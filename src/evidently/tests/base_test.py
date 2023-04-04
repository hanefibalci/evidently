import abc
import dataclasses
from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

from pydantic import BaseModel
from pydantic import Field

from evidently.base_metric import MetricResult
from evidently.utils.generators import BaseGenerator
from evidently.utils.generators import make_generator_by_columns
from evidently.utils.types import ApproxValue
from evidently.utils.types import Numeric
from evidently.utils.types import NumericApprox

if TYPE_CHECKING:
    from pydantic.typing import DictStrAny

    from evidently.suite.base_suite import Context


@dataclasses.dataclass
class GroupData:
    id: str
    title: str
    description: str
    sort_index: int = 0
    severity: Optional[str] = None


@dataclasses.dataclass
class GroupTypeData:
    id: str
    title: str
    # possible values with description, if empty will use simple view (no severity, description and sorting).
    values: List[GroupData] = dataclasses.field(default_factory=list)

    def add_value(self, data: GroupData):
        self.values.append(data)


class GroupingTypes:
    ByFeature = GroupTypeData(
        "by_feature",
        "By feature",
        [
            GroupData(
                "no group",
                "Dataset-level tests",
                "Some tests cannot be grouped by feature",
            )
        ],
    )
    ByClass = GroupTypeData("by_class", "By class", [])
    TestGroup = GroupTypeData(
        "test_group",
        "By test group",
        [
            GroupData(
                "no group",
                "Ungrouped",
                "Some tests don’t belong to any group under the selected condition",
            )
        ],
    )
    TestType = GroupTypeData("test_type", "By test type", [])


DEFAULT_GROUP = [
    GroupingTypes.ByFeature,
    GroupingTypes.TestGroup,
    GroupingTypes.TestType,
    GroupingTypes.ByClass,
]


class EnumValueMixin(BaseModel):
    def dict(self, *args, **kwargs) -> "DictStrAny":
        res = super().dict(*args, **kwargs)
        return {k: v.value if isinstance(v, Enum) else v for k, v in res.items()}


class ExcludeNoneMixin(BaseModel):
    def dict(self, *args, **kwargs) -> "DictStrAny":
        kwargs["exclude_none"] = True
        return super().dict(*args, **kwargs)


class TestStatus(Enum):
    # Constants for test result status
    SUCCESS = "SUCCESS"  # the test was passed
    FAIL = "FAIL"  # success pass for the test
    WARNING = "WARNING"  # the test was passed, but we have some issues during the execution
    ERROR = "ERROR"  # cannot calculate the test result, no data
    SKIPPED = "SKIPPED"  # the test was skipped


class TestParameters(MetricResult):
    pass


class TestResult(EnumValueMixin, MetricResult):  # todo: create common base class
    # short name/title from the test class
    name: str
    # what was checked, what threshold (current value 13 is not ok with condition less than 5)
    description: str
    # status of the test result
    status: TestStatus
    # grouping parameters
    group: str
    groups: Dict[str, str] = Field(default_factory=dict, exclude=True)
    parameters: Optional[TestParameters]
    exception: Optional[BaseException] = Field(None, exclude=True)

    def set_status(self, status: TestStatus, description: Optional[str] = None) -> None:
        self.status = status

        if description is not None:
            self.description = description

    def mark_as_fail(self, description: Optional[str] = None):
        self.set_status(TestStatus.FAIL, description=description)

    def mark_as_error(self, description: Optional[str] = None):
        self.set_status(TestStatus.ERROR, description=description)

    def mark_as_success(self, description: Optional[str] = None):
        self.set_status(TestStatus.SUCCESS, description=description)

    def mark_as_warning(self, description: Optional[str] = None):
        self.set_status(TestStatus.WARNING, description=description)

    def is_passed(self):
        return self.status in [TestStatus.SUCCESS, TestStatus.WARNING]


class Test:
    """
    all fields in test class with type that is subclass of Metric would be used as dependencies of test.
    """

    name: str
    group: str
    context: "Context" = None

    @abc.abstractmethod
    def check(self) -> TestResult:
        raise NotImplementedError

    def set_context(self, context: "Context"):
        self.context = context

    def get_result(self) -> TestResult:
        if self.context is None:
            raise ValueError("No context is set")
        result = self.context.test_results.get(self, None)
        if result is None:
            raise ValueError(f"No result found for metric {self} of type {type(self).__name__}")
        return result


class ValueSource(Enum):
    USER = "user"
    CURRENT = "current"
    REFERENCE = "reference"
    DUMMY = "dummy"
    OTHER = "other"


class TestValueCondition(ExcludeNoneMixin):
    """
    Class for processing a value conditions - should it be less, greater than, equals and so on.

    An object of the class stores specified conditions and can be used for checking a value by them.
    """

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        smart_union = True

    eq: Optional[NumericApprox] = None
    gt: Optional[NumericApprox] = None
    gte: Optional[NumericApprox] = None
    is_in: Optional[List[Union[Numeric, str, bool]]] = None
    lt: Optional[NumericApprox] = None
    lte: Optional[NumericApprox] = None
    not_eq: Optional[Numeric] = None
    not_in: Optional[List[Union[Numeric, str, bool]]] = None
    source: Optional[ValueSource] = Field(None, exclude=True)  # todo: temporary to not fix tests

    def has_condition(self) -> bool:
        """
        Checks if we have a condition in the object and returns True in this case.

        If we have no conditions - returns False.
        """
        return any(
            value is not None
            for value in (
                self.eq,
                self.gt,
                self.gte,
                self.is_in,
                self.lt,
                self.lte,
                self.not_in,
                self.not_eq,
            )
        )

    def check_value(self, value: Numeric) -> bool:
        result = True

        if self.eq is not None and result:
            result = value == self.eq

        if self.gt is not None and result:
            result = value > self.gt

        if self.gte is not None and result:
            result = value >= self.gte

        if self.is_in is not None and result:
            result = value in self.is_in

        if self.lt is not None and result:
            result = value < self.lt

        if self.lte is not None and result:
            result = value <= self.lte

        if self.not_eq is not None and result:
            result = value != self.not_eq

        if self.not_in is not None and result:
            result = value not in self.not_in

        return result

    def __str__(self) -> str:
        conditions = []
        operations = ["eq", "gt", "gte", "lt", "lte", "not_eq", "is_in", "not_in"]

        for op in operations:
            value = getattr(self, op)

            if value is None:
                continue

            if isinstance(value, (float, ApproxValue)):
                conditions.append(f"{op}={value:.3g}")

            else:
                conditions.append(f"{op}={value}")

        return f"{' and '.join(conditions)}"


class ConditionTestParameters(TestParameters):
    condition: TestValueCondition


class BaseConditionsTest(Test, ABC):
    """
    Base class for all tests with a condition
    """

    condition: TestValueCondition

    def __init__(
        self,
        eq: Optional[NumericApprox] = None,
        gt: Optional[Numeric] = None,
        gte: Optional[Numeric] = None,
        is_in: Optional[List[Union[Numeric, str, bool]]] = None,
        lt: Optional[Numeric] = None,
        lte: Optional[Numeric] = None,
        not_eq: Optional[Numeric] = None,
        not_in: Optional[List[Union[Numeric, str, bool]]] = None,
    ):
        self.condition = TestValueCondition(
            eq=eq,
            gt=gt,
            gte=gte,
            is_in=is_in,
            lt=lt,
            lte=lte,
            not_eq=not_eq,
            not_in=not_in,
        )


class CheckValueParameters(ConditionTestParameters):
    value: Optional[Numeric]


class ColumnCheckValueParameters(CheckValueParameters):
    column_name: str


class BaseCheckValueTest(BaseConditionsTest):
    """
    Base class for all tests with checking a value condition
    """

    value: Numeric

    @abc.abstractmethod
    def calculate_value_for_test(self) -> Optional[Any]:
        """Method for getting the checking value.

        Define it in a child class"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_description(self, value: Numeric) -> str:
        """Method for getting a description that we can use.
        The description can use the checked value.

        Define it in a child class"""
        raise NotImplementedError()

    def get_condition(self) -> TestValueCondition:
        return self.condition

    def groups(self) -> Dict[str, str]:
        return {}

    def get_parameters(self) -> CheckValueParameters:
        return CheckValueParameters(condition=self.get_condition(), value=self.value)

    def check(self):
        result = TestResult(
            name=self.name,
            description="The test was not launched",
            status=TestStatus.SKIPPED,
            group=self.group,
            parameters=None,
        )
        value = self.calculate_value_for_test()
        self.value = value
        result.description = self.get_description(value)
        result.parameters = self.get_parameters()

        try:
            if value is None:
                result.mark_as_error()

            else:
                condition = self.get_condition()

                if condition is None:
                    raise ValueError

                condition_check_result = condition.check_value(value)

                if condition_check_result:
                    result.mark_as_success()

                else:
                    result.mark_as_fail()

        except ValueError:
            result.mark_as_error("Cannot calculate the condition")

        result.groups.update(self.groups())
        return result


def generate_column_tests(
    test_class: Type[Test], columns: Optional[Union[str, list]] = None, parameters: Optional[Dict] = None
) -> BaseGenerator:
    """Function for generating tests for columns"""
    return make_generator_by_columns(
        base_class=test_class,
        columns=columns,
        parameters=parameters,
    )

from typing import List
from typing import Optional
from typing import Union

import numpy as np

from evidently.base_metric import ColumnName
from evidently.base_metric import InputData
from evidently.base_metric import Metric
from evidently.base_metric import MetricResult
from evidently.core import ColumnType
from evidently.metric_results import Distribution
from evidently.model.widget import BaseWidgetInfo
from evidently.renderers.base_renderer import MetricRenderer
from evidently.renderers.base_renderer import default_renderer
from evidently.renderers.html_widgets import WidgetSize
from evidently.renderers.html_widgets import header_text
from evidently.renderers.html_widgets import plotly_figure
from evidently.renderers.render_utils import get_distribution_plot_figure
from evidently.utils.visualizations import get_distribution_for_column


class ColumnDistributionMetricResult(MetricResult):
    column_name: str
    current: Distribution
    reference: Optional[Distribution] = None


class ColumnDistributionMetric(Metric[ColumnDistributionMetricResult]):
    """Calculates distribution for the column"""

    column_name: ColumnName

    def __init__(self, column_name: Union[str, ColumnName]) -> None:
        if isinstance(column_name, str):
            self.column_name = ColumnName.main_dataset(column_name)
        else:
            self.column_name = column_name

    def calculate(self, data: InputData) -> ColumnDistributionMetricResult:
        if not data.has_column(self.column_name):
            raise ValueError(f"Column '{self.column_name.display_name}' was not found in data.")

        if not self.column_name.is_main_dataset():
            column_type = ColumnType.Numerical
        else:
            column_type = data.data_definition.get_column(self.column_name.name).column_type
        current_column = data.get_current_column(self.column_name).replace([np.inf, -np.inf], np.nan)
        reference_column = data.get_reference_column(self.column_name)
        if reference_column is not None:
            reference_column = reference_column.replace([np.inf, -np.inf], np.nan)
        current, reference = get_distribution_for_column(
            column_type=column_type.value,
            current=current_column,
            reference=reference_column,
        )

        return ColumnDistributionMetricResult(
            column_name=self.column_name.display_name,
            current=current,
            reference=reference,
        )


@default_renderer(wrap_type=ColumnDistributionMetric)
class ColumnDistributionMetricRenderer(MetricRenderer):
    def render_html(self, obj: ColumnDistributionMetric) -> List[BaseWidgetInfo]:
        metric_result = obj.get_result()
        distr_fig = get_distribution_plot_figure(
            current_distribution=metric_result.current,
            reference_distribution=metric_result.reference,
            color_options=self.color_options,
        )

        result = [
            header_text(label=f"Distribution for column '{metric_result.column_name}'."),
            plotly_figure(title="", figure=distr_fig, size=WidgetSize.FULL),
        ]
        return result

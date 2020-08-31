"""
Metrics related to order predictions
"""
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric

from parlai_diplomacy.utils.game_to_sequence_formatting import (
    order_seq_to_fairdip,
    order_is_empty,
    all_orders_seq_to_dct,
)

from typing import Optional, Tuple


class OrderPredMetricMixin:
    """
    Mixin to add metrics to teachers which return a single order prediction.
    """

    def custom_evaluation(
        self, teacher_action: Message, labels: Optional[Tuple[str]], model_response: Message,
    ) -> None:
        if "text" not in model_response:
            # model didn't speak, skip this example
            return
        order_label = set(order_seq_to_fairdip(labels[0]))
        order_pred = set(order_seq_to_fairdip(model_response["text"]))
        empty_order = order_is_empty(labels[0])

        if order_label == order_pred:
            # count 1 / 1 exact match
            self.metrics.add("order_exact_avg", AverageMetric(1, 1))
            if not empty_order:
                self.metrics.add("order_exact_no_empty_avg", AverageMetric(1, 1))
        else:
            # count 0 /1 exact match
            self.metrics.add("order_exact_avg", AverageMetric(0, 1))
            if not empty_order:
                self.metrics.add("order_exact_no_empty_avg", AverageMetric(0, 1))


class AllOrderPredMetricMixin:
    """
    Mixin to add metrics to teachers which return predictions for all orders
    """

    def custom_evaluation(
        self, teacher_action: Message, labels: Optional[Tuple[str]], model_response: Message,
    ) -> None:
        if "text" not in model_response:
            # model didn't speak, skip this example
            return

        orders_label = all_orders_seq_to_dct(labels[0])
        orders_pred = all_orders_seq_to_dct(model_response["text"])
        player = teacher_action["player"]

        for power, order_label in orders_label.items():
            order_pred = orders_pred.get(power)
            if order_pred is None:
                self.metrics.add("all_order_exact_avg", AverageMetric(0, 1))
                continue

            empty_order = not order_label

            if order_label == order_pred:
                # count 1 / 1 exact match
                self.metrics.add("all_order_exact_avg", AverageMetric(1, 1))
                if not empty_order:
                    self.metrics.add("all_order_exact_no_empty_avg", AverageMetric(1, 1))
                if power == player:
                    # get metrics for predicting your OWN order
                    self.metrics.add("order_exact_avg", AverageMetric(1, 1))
                    if not empty_order:
                        self.metrics.add("order_exact_no_empty_avg", AverageMetric(1, 1))
            else:
                # count 0 / 1 exact match
                self.metrics.add("all_order_exact_avg", AverageMetric(0, 1))
                if not empty_order:
                    self.metrics.add("all_order_exact_no_empty_avg", AverageMetric(0, 1))
                if power == player:
                    # get metrics for predicting your OWN order
                    self.metrics.add("order_exact_avg", AverageMetric(0, 1))
                    if not empty_order:
                        self.metrics.add("order_exact_no_empty_avg", AverageMetric(0, 1))

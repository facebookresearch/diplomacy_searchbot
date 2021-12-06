from typing import Any, Tuple, Dict, List

Power = str
Action = Tuple[str, ...]  # a set of orders
Phase = str
GameJson = Dict[str, Any]
JointAction = Dict[Power, Action]
JointActionValues = Dict[Power, float]

RolloutResult = Tuple[JointAction, JointActionValues]
RolloutResults = List[RolloutResult]

Policy = Dict[Action, float]
PowerPolicies = Dict[Power, Policy]
JointPolicy = List[Tuple[JointAction, float]]
PlausibleOrders = Dict[Power, List[Action]]

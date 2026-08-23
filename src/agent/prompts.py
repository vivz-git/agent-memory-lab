"""Prompt templates and formatters for RegAgent and CIC-IoT Agent."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.environments.base import TaskQuery
    from src.agent.core import Demonstration

# ==============================================================================
# RegAgent Prompts (Appendix A.2)
# ==============================================================================

REGAGENT_PROMPT_TEMPLATE = """You are given a 6-dimensional input vector x. Predict y = w^T x with an unknown w.
You will see K demonstrations of (input, guess) pairs that use the same w but may contain noise in all demonstrations.
You need to strictly follow the output content and format of the demonstrations, which is Guess:
boxed{{<number>}} without any other text.
Demonstrations (K={k}):
{demonstrations}
Now solve for the new input.
Input: {x} Guess:"""

REGAGENT_DEMO_TEMPLATE = "Input: {x} Guess: boxed{{{y}}}"


def format_regagent_prompt(
    query: TaskQuery,
    demonstrations: List[Demonstration],
) -> str:
    """Format RegAgent prompt strictly following Appendix A.2."""
    demos_text_list = []
    for d in demonstrations:
        if isinstance(d.query.raw_input, (list, np.ndarray)):
            x_arr = np.asarray(d.query.raw_input, dtype=np.float64)
            x_str = "[" + ", ".join(f"{v:.4f}" for v in x_arr) + "]"
        else:
            x_str = str(d.query.raw_input)

        exec_str = d.execution.strip()
        if not exec_str.startswith("boxed{") and not exec_str.startswith(r"\boxed{"):
            exec_str = f"boxed{{{exec_str}}}"

        demos_text_list.append(f"Input: {x_str} Guess: {exec_str}")

    demos_formatted = "\n".join(demos_text_list) if demos_text_list else "None"

    if isinstance(query.raw_input, (list, np.ndarray)):
        curr_x_arr = np.asarray(query.raw_input, dtype=np.float64)
        curr_x_str = "[" + ", ".join(f"{v:.4f}" for v in curr_x_arr) + "]"
    else:
        curr_x_str = str(query.raw_input)

    return REGAGENT_PROMPT_TEMPLATE.format(
        k=len(demonstrations),
        demonstrations=demos_formatted,
        x=curr_x_str,
    )


# ==============================================================================
# CIC-IoT Prompts (Appendix A.3)
# ==============================================================================

CICIOT_PROMPT_TEMPLATE = """Based on the following features, determine the most likely traffic type from the list below:
* Pay special attention to cross-field consistency checks.
* Do not be misled by a single feature if it conflicts with others.
* Note: Your reasoning should be based on all features, not on any single field. You are allowed to select only one traffic type as your answer. If you choose more than one, your answer will be marked as incorrect.
Required output format:
ANALYSIS: {{your reasoning here, including key features and justification}}
ANSWER: {{traffic_type}}

{features_block}

### Traffic Types:
['DDoS-ICMP_Flood', 'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood', 'DDoS-HTTP_Flood', 'BenignTraffic']"""

CICIOT_WITH_DEMOS_TEMPLATE = """You are given network traffic flow features and need to classify the traffic type.
You will see K demonstrations of (features, analysis, answer) pairs.

Demonstrations (K={k}):
{demonstrations}

Now classify the new traffic flow:
Based on the following features, determine the most likely traffic type from the list below:
* Pay special attention to cross-field consistency checks.
* Do not be misled by a single feature if it conflicts with others.
* Note: Your reasoning should be based on all features, not on any single field. You are allowed to select only one traffic type as your answer. If you choose more than one, your answer will be marked as incorrect.
Required output format:
ANALYSIS: {{your reasoning here, including key features and justification}}
ANSWER: {{traffic_type}}

{features_block}

### Traffic Types:
['DDoS-ICMP_Flood', 'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood', 'DDoS-HTTP_Flood', 'BenignTraffic']"""


def format_ciciot_features_block(features: Dict[str, Any]) -> str:
    """Format dictionary of 33+ flow features per Appendix A.3 specification."""
    lines = [
        f"Flow duration [description: Duration of the packet's flow]: {features.get('flow_duration', 0.0)}",
        f"Header Length [description: Header Length]: {features.get('Header_Length', 0)} bytes",
        f"Protocol Type [description: IP, UDP, TCP, IGMP, ICMP, Unknown (Integers)]: {features.get('Protocol_Type', 0)}",
        f"Duration [description: Time-to-Live (ttl)]: {features.get('Duration', 0)}",
        f"Rate [description: Rate of packet transmission in a flow]: {features.get('Rate', 0.0)}",
        f"Srate [description: Rate of outbound packets transmission in a flow]: {features.get('Srate', 0.0)}",
        f"Drate [description: Rate of inbound packets transmission in a flow]: {features.get('Drate', 0.0)}",
        f"Number of FIN flags [description: FIN flag value]: {features.get('fin_flag_number', 0)}",
        f"Number of SYN flags [description: SYN flag value]: {features.get('syn_flag_number', 0)}",
        f"Number of RST flags [description: RST flag value]: {features.get('rst_flag_number', 0)}",
        f"Number of PSH flags [description: PSH flag value]: {features.get('psh_flag_number', 0)}",
        f"Number of ACK flags [description: ACK flag value]: {features.get('ack_flag_number', 0)}",
        f"Number of ECE flags [description: ECE flag value]: {features.get('ece_flag_number', 0)}",
        f"Number of CWR flags [description: CWR flag value]: {features.get('cwr_flag_number', 0)}",
        f"Number of ACK packets: {features.get('ack_count', 0)}",
        f"Number of SYN packets: {features.get('syn_count', 0)}",
        f"Number of FIN packets: {features.get('fin_count', 0)}",
        f"Number of URG packets: {features.get('urg_count', 0)}",
        f"Number of RST packets: {features.get('rst_count', 0)}",
        f"HTTP traffic flag: {features.get('HTTP', 0)}",
        f"HTTPS traffic flag: {features.get('HTTPS', 0)}",
        f"DNS traffic flag: {features.get('DNS', 0)}",
        f"Telnet traffic flag: {features.get('Telnet', 0)}",
        f"SMTP traffic flag: {features.get('SMTP', 0)}",
        f"SSH traffic flag: {features.get('SSH', 0)}",
        f"IRC traffic flag: {features.get('IRC', 0)}",
        f"TCP traffic flag: {features.get('TCP', 0)}",
        f"UDP traffic flag: {features.get('UDP', 0)}",
        f"DHCP traffic flag: {features.get('DHCP', 0)}",
        f"ARP traffic flag: {features.get('ARP', 0)}",
        f"ICMP traffic flag: {features.get('ICMP', 0)}",
        f"IPv4 traffic flag: {features.get('IPv', 1)}",
        f"LLC traffic flag: {features.get('LLC', 1)}",
        f"Total sum of feature values: {features.get('Tot_sum', 0.0)}",
        f"Minimum value: {features.get('Min', 0.0)}",
        f"Maximum value: {features.get('Max', 0.0)}",
        f"Average value: {features.get('AVG', 0.0)}",
        f"Standard deviation: {features.get('Std', 0.0)}",
        f"Total size of the flow: {features.get('Tot_size', 0.0)} bytes",
        f"Inter-arrival time (milliseconds): {features.get('IAT', 0.0)}",
        f"Number of packets or flows: {features.get('Number', 0.0)}",
        f"Magnitude of the flow: {features.get('Magnitude', 0.0)}",
        f"Radius of the flow: {features.get('Radius', 0.0)}",
        f"Covariance of the flow: {features.get('Covariance', 0.0)}",
        f"Variance of the flow: {features.get('Variance', 0.0)}",
        f"Weight of the flow: {features.get('Weight', 0.0)}",
    ]
    return "\n".join(lines)


def format_ciciot_prompt(
    query: TaskQuery,
    demonstrations: List[Demonstration],
) -> str:
    """Format CIC-IoT prompt with demonstrations."""
    curr_features = query.features if query.features else query.raw_input
    curr_block = format_ciciot_features_block(curr_features)

    if not demonstrations:
        return CICIOT_PROMPT_TEMPLATE.format(features_block=curr_block)

    demos_formatted_list = []
    for i, d in enumerate(demonstrations):
        d_feats = d.query.features if d.query.features else d.query.raw_input
        d_block = format_ciciot_features_block(d_feats)
        d_exec = d.execution.strip()
        demos_formatted_list.append(
            f"--- Demonstration {i+1} ---\n{d_block}\n{d_exec}"
        )

    demos_text = "\n\n".join(demos_formatted_list)

    return CICIOT_WITH_DEMOS_TEMPLATE.format(
        k=len(demonstrations),
        demonstrations=demos_text,
        features_block=curr_block,
    )


# ==============================================================================
# Evaluator Prompts (Appendix A.3)
# ==============================================================================

CICIOT_COARSE_EVALUATOR_TEMPLATE = """You are a strict evaluator for IoT traffic classification answers. You should infer the most likely label from the provided flow-level features, then judge whether the model's answer is CORRECT or INCORRECT.
### Judging Criteria: The Model Answer is CORRECT if its predicted label exactly matches your inferred gold label, else INCORRECT.
### Judging Steps:
1) Carefully analyze key fields and reason your inferred gold label.
2) Compare the Model Answer's label to your gold label.
### Problem: Based on the following features, determine the most likely traffic type from the list below:
### Features: {problem}
### Model Answer: {generated_answer}
- Respond with your judgement and explanation as following format:
- First line: Respond with 'CORRECT' or 'INCORRECT' only.
- Following lines: Provide your reasoning or chain-of-thought.
Your judgement:"""

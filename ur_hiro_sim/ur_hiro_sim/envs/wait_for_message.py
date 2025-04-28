from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec


def wait_for_message(
    msg_type,
    node: 'Node',
    topic: str,
    time_to_wait=-1
):
    """
    Wait for the next incoming message.
    :param msg_type: message type
    :param node: node to initialize the subscription on
    :param topic: topic name to wait for message
    :time_to_wait: seconds to wait before returning
    :return (True, msg) if a message was successfully received, (False, ()) if message
        could not be obtained or shutdown was triggered asynchronously on the context.
    """
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()

    sub = node.create_subscription(msg_type, topic, lambda _: None, 1)
    wait_set.add_subscription(sub.handle)
    sigint_gc = SignalHandlerGuardCondition(context=context)
    wait_set.add_guard_condition(sigint_gc.handle)

    timeout_nsec = timeout_sec_to_nsec(time_to_wait)
    wait_set.wait(timeout_nsec)

    subs_ready = wait_set.get_ready_entities('subscription')
    guards_ready = wait_set.get_ready_entities('guard_condition')

    if guards_ready:
        if sigint_gc.handle.pointer in guards_ready:
            return (False, None)

    if subs_ready:
        if sub.handle.pointer in subs_ready:
            msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
            return (True, msg_info[0])

    return (False, None)
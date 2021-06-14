# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from scout_msgs/ScoutStatus.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import scout_msgs.msg
import std_msgs.msg

class ScoutStatus(genpy.Message):
  _md5sum = "fef327f257c6c79882a0f333c289c7dd"
  _type = "scout_msgs/ScoutStatus"
  _has_header = True  # flag to mark the presence of a Header object
  _full_text = """Header header

int8 MOTOR_ID_FRONT_RIGHT = 0
int8 MOTOR_ID_FRONT_LEFT = 1
int8 MOTOR_ID_REAR_RIGHT = 2
int8 MOTOR_ID_REAR_LEFT = 3

int8 LIGHT_ID_FRONT = 0
int8 LIGHT_ID_REAR = 1

# motion state
float64 linear_velocity
float64 angular_velocity
float64 transverse_linear_velocity

# base state
uint8 base_state
uint8 control_mode
uint16 fault_code
float64 battery_voltage

# motor state
ScoutMotorState[4] motor_states

# light state
bool light_control_enabled
ScoutLightState front_light_state
ScoutLightState rear_light_state

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: scout_msgs/ScoutMotorState
float64 current
float64 rpm
float64 temperature
================================================================================
MSG: scout_msgs/ScoutLightState
uint8 mode
uint8 custom_value"""
  # Pseudo-constants
  MOTOR_ID_FRONT_RIGHT = 0
  MOTOR_ID_FRONT_LEFT = 1
  MOTOR_ID_REAR_RIGHT = 2
  MOTOR_ID_REAR_LEFT = 3
  LIGHT_ID_FRONT = 0
  LIGHT_ID_REAR = 1

  __slots__ = ['header','linear_velocity','angular_velocity','transverse_linear_velocity','base_state','control_mode','fault_code','battery_voltage','motor_states','light_control_enabled','front_light_state','rear_light_state']
  _slot_types = ['std_msgs/Header','float64','float64','float64','uint8','uint8','uint16','float64','scout_msgs/ScoutMotorState[4]','bool','scout_msgs/ScoutLightState','scout_msgs/ScoutLightState']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       header,linear_velocity,angular_velocity,transverse_linear_velocity,base_state,control_mode,fault_code,battery_voltage,motor_states,light_control_enabled,front_light_state,rear_light_state

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(ScoutStatus, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.linear_velocity is None:
        self.linear_velocity = 0.
      if self.angular_velocity is None:
        self.angular_velocity = 0.
      if self.transverse_linear_velocity is None:
        self.transverse_linear_velocity = 0.
      if self.base_state is None:
        self.base_state = 0
      if self.control_mode is None:
        self.control_mode = 0
      if self.fault_code is None:
        self.fault_code = 0
      if self.battery_voltage is None:
        self.battery_voltage = 0.
      if self.motor_states is None:
        self.motor_states = [scout_msgs.msg.ScoutMotorState() for _ in range(4)]
      if self.light_control_enabled is None:
        self.light_control_enabled = False
      if self.front_light_state is None:
        self.front_light_state = scout_msgs.msg.ScoutLightState()
      if self.rear_light_state is None:
        self.rear_light_state = scout_msgs.msg.ScoutLightState()
    else:
      self.header = std_msgs.msg.Header()
      self.linear_velocity = 0.
      self.angular_velocity = 0.
      self.transverse_linear_velocity = 0.
      self.base_state = 0
      self.control_mode = 0
      self.fault_code = 0
      self.battery_voltage = 0.
      self.motor_states = [scout_msgs.msg.ScoutMotorState() for _ in range(4)]
      self.light_control_enabled = False
      self.front_light_state = scout_msgs.msg.ScoutLightState()
      self.rear_light_state = scout_msgs.msg.ScoutLightState()

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_3d2BHd().pack(_x.linear_velocity, _x.angular_velocity, _x.transverse_linear_velocity, _x.base_state, _x.control_mode, _x.fault_code, _x.battery_voltage))
      if len(self.motor_states) != 4:
        self._check_types(ValueError("Expecting %s items but found %s when writing '%s'" % (4, len(self.motor_states), 'self.motor_states')))
      for val1 in self.motor_states:
        _x = val1
        buff.write(_get_struct_3d().pack(_x.current, _x.rpm, _x.temperature))
      _x = self
      buff.write(_get_struct_5B().pack(_x.light_control_enabled, _x.front_light_state.mode, _x.front_light_state.custom_value, _x.rear_light_state.mode, _x.rear_light_state.custom_value))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.motor_states is None:
        self.motor_states = None
      if self.front_light_state is None:
        self.front_light_state = scout_msgs.msg.ScoutLightState()
      if self.rear_light_state is None:
        self.rear_light_state = scout_msgs.msg.ScoutLightState()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 36
      (_x.linear_velocity, _x.angular_velocity, _x.transverse_linear_velocity, _x.base_state, _x.control_mode, _x.fault_code, _x.battery_voltage,) = _get_struct_3d2BHd().unpack(str[start:end])
      self.motor_states = []
      for i in range(0, 4):
        val1 = scout_msgs.msg.ScoutMotorState()
        _x = val1
        start = end
        end += 24
        (_x.current, _x.rpm, _x.temperature,) = _get_struct_3d().unpack(str[start:end])
        self.motor_states.append(val1)
      _x = self
      start = end
      end += 5
      (_x.light_control_enabled, _x.front_light_state.mode, _x.front_light_state.custom_value, _x.rear_light_state.mode, _x.rear_light_state.custom_value,) = _get_struct_5B().unpack(str[start:end])
      self.light_control_enabled = bool(self.light_control_enabled)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_3d2BHd().pack(_x.linear_velocity, _x.angular_velocity, _x.transverse_linear_velocity, _x.base_state, _x.control_mode, _x.fault_code, _x.battery_voltage))
      if len(self.motor_states) != 4:
        self._check_types(ValueError("Expecting %s items but found %s when writing '%s'" % (4, len(self.motor_states), 'self.motor_states')))
      for val1 in self.motor_states:
        _x = val1
        buff.write(_get_struct_3d().pack(_x.current, _x.rpm, _x.temperature))
      _x = self
      buff.write(_get_struct_5B().pack(_x.light_control_enabled, _x.front_light_state.mode, _x.front_light_state.custom_value, _x.rear_light_state.mode, _x.rear_light_state.custom_value))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.motor_states is None:
        self.motor_states = None
      if self.front_light_state is None:
        self.front_light_state = scout_msgs.msg.ScoutLightState()
      if self.rear_light_state is None:
        self.rear_light_state = scout_msgs.msg.ScoutLightState()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 36
      (_x.linear_velocity, _x.angular_velocity, _x.transverse_linear_velocity, _x.base_state, _x.control_mode, _x.fault_code, _x.battery_voltage,) = _get_struct_3d2BHd().unpack(str[start:end])
      self.motor_states = []
      for i in range(0, 4):
        val1 = scout_msgs.msg.ScoutMotorState()
        _x = val1
        start = end
        end += 24
        (_x.current, _x.rpm, _x.temperature,) = _get_struct_3d().unpack(str[start:end])
        self.motor_states.append(val1)
      _x = self
      start = end
      end += 5
      (_x.light_control_enabled, _x.front_light_state.mode, _x.front_light_state.custom_value, _x.rear_light_state.mode, _x.rear_light_state.custom_value,) = _get_struct_5B().unpack(str[start:end])
      self.light_control_enabled = bool(self.light_control_enabled)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_3I = None
def _get_struct_3I():
    global _struct_3I
    if _struct_3I is None:
        _struct_3I = struct.Struct("<3I")
    return _struct_3I
_struct_3d = None
def _get_struct_3d():
    global _struct_3d
    if _struct_3d is None:
        _struct_3d = struct.Struct("<3d")
    return _struct_3d
_struct_3d2BHd = None
def _get_struct_3d2BHd():
    global _struct_3d2BHd
    if _struct_3d2BHd is None:
        _struct_3d2BHd = struct.Struct("<3d2BHd")
    return _struct_3d2BHd
_struct_5B = None
def _get_struct_5B():
    global _struct_5B
    if _struct_5B is None:
        _struct_5B = struct.Struct("<5B")
    return _struct_5B
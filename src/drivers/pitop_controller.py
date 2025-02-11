from math import floor
import math

from pitop import BrakingType, EncoderMotor, ForwardDirection, Pitop
from pitop.pma.common.encoder_motor_registers import MotorSyncBits, MotorSyncRegisters
from pitop.pma.plate_interface import PlateInterface
from pitop.common.firmware_device import FirmwareDevice, FirmwareDeviceID


class PitopController:
    def __init__(
        self,
        wheel_separation,
        wheel_diameter,
        left_port="M3",
        right_port="M0",
        invert_left=False,
        invert_right=False,
    ):
        self.ready = False

        self._pitop = Pitop()
        self.miniscreen = self._pitop.miniscreen

        self.left_motor_port = left_port
        self.right_motor_port = right_port

        # chassis setup
        self.wheel_separation = wheel_separation
        self.wheel_diameter = wheel_diameter

        left_direction = ForwardDirection.CLOCKWISE
        right_direction = ForwardDirection.COUNTER_CLOCKWISE
        if invert_left:
            left_direction = ForwardDirection.COUNTER_CLOCKWISE
        if invert_right:
            right_direction = ForwardDirection.CLOCKWISE

        # Setup the motors
        self.left_motor = EncoderMotor(
            left_port,
            forward_direction=left_direction,
            braking_type=BrakingType.COAST,
            wheel_diameter=self.wheel_diameter,
            name="Left Motor",
        )

        self.right_motor = EncoderMotor(
            right_port,
            forward_direction=right_direction,
            braking_type=BrakingType.COAST,
            wheel_diameter=self.wheel_diameter,
            name="Right Motor",
        )

        # Round down to ensure no speed value ever goes above maximum due to rounding issues
        # (resulting in error)
        self.max_motor_speed = (
            min(self.left_motor.max_rpm, self.right_motor.max_rpm) * math.pi / 30.0
        )

        print("Max motor speed: ", self.max_motor_speed, "rad/s")

        # Motor syncing
        self.__mcu_device = PlateInterface().get_device_mcu()
        self._set_synchronous_motor_movement_mode()

        # make sure the Expansion Plate is attached to the pi-top
        device = FirmwareDevice(FirmwareDeviceID.pt4_expansion_plate)
        version = device.get_sch_hardware_version_major()
        if version is None:
            raise Exception("Expansion Plate not found")

        self.ready = True

    def _set_synchronous_motor_movement_mode(self):
        sync_config = (
            MotorSyncBits[self.left_motor_port].value
            | MotorSyncBits[self.right_motor_port].value
        )
        self.__mcu_device.write_byte(MotorSyncRegisters.CONFIG.value, sync_config)

    def _start_synchronous_motor_movement(self):
        self.__mcu_device.write_byte(MotorSyncRegisters.START.value, 1)

    def current_speed(self):
        if not self.ready:
            return 0.0, 0.0
        left_speed = self.left_motor.current_rpm * math.pi / 30.0
        right_speed = self.right_motor.current_rpm * math.pi / 30.0
        return left_speed, right_speed

    def robot_move(self, speed_left, speed_right):
        if not self.ready:
            return
        rpm_left = speed_left * 30.0 / math.pi
        rpm_right = speed_right * 30.0 / math.pi
        self.left_motor.set_target_rpm(rpm_left)
        self.right_motor.set_target_rpm(rpm_right)
        self._start_synchronous_motor_movement()

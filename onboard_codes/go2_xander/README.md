# Code for deploying Go2

## Code Structure

- go2_xander
  - go2_run.py: Publish Go2Node for controlling Go2
  - go2_visual.py: Publish fisheye camera node @Tianhang Wu
  - unitree_ros2_real.py: UnitreeRos2Real

## Unitree Go2 Msg

- WirelessController
```
float32 lx
float32 ly
float32 rx
float32 ry
uint16 keys
```

- LowState
```
uint8[2] head
uint8 level_flag
uint8 frame_reserve
uint32[2] sn
uint32[2] version
uint16 bandwidth
IMUState imu_state
MotorState[20] motor_state
BmsState bms_state
int16[4] foot_force
int16[4] foot_force_est
uint32 tick
uint8[40] wireless_remote
uint8 bit_flag
float32 adc_reel
int8 temperature_ntc1
int8 temperature_ntc2
float32 power_v
float32 power_a
uint16[4] fan_frequency
uint32 reserve
uint32 crc
```

- LowCmd
```
uint8[2] head
uint8 level_flag
uint8 frame_reserve
uint32[2] sn
uint32[2] version
uint16 bandwidth
MotorCmd[20] motor_cmd
BmsCmd bms_cmd
uint8[40] wireless_remote
uint8[12] led
uint8[2] fan
uint8 gpio
uint32 reserve
uint32 crc
```

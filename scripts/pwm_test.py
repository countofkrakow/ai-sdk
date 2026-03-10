from periphery import GPIO, PWM
import time


def pwm_test():
    pwm = None
    gate = None

    try:
        # Configure PWM parameters - modify according to your hardware
        # Common PWM paths: /sys/class/pwm/pwmchip0/pwm0, etc.
        PWM_CHIP = 10       # PWM chip number
        PWM_CHANNEL = 2     # PWM channel number

        # MOSFET gate control on pin 32 (PD0)
        GATE_CHIP = "/dev/gpiochip0"
        GATE_LINE = 0

        # Initialize PWM
        pwm = PWM(PWM_CHIP, PWM_CHANNEL)

        # Initialize MOSFET gate control pin
        gate = GPIO(GATE_CHIP, GATE_LINE, "out")
        gate.write(True)

        print(f"PWM Test: Using PWM{PWM_CHIP}.{PWM_CHANNEL} (Pin: PD12)")
        print("MOSFET gate enabled on pin 32 (PD0)")

        # Set PWM frequency to 1kHz
        frequency = 1000  # 1000 Hz
        pwm.frequency = frequency
        print(f"  Set frequency: {frequency} Hz")

        # Enable PWM output
        pwm.enable()
        print("  PWM enabled")

        # Test different duty cycles (0% to 100%)
        duty_cycles = [0.0, 0.25, 0.5, 0.75, 1.0]

        for duty in duty_cycles:
            pwm.duty_cycle = duty
            print(f"  Set duty cycle: {duty*100:.1f}%")
            time.sleep(1)  # Maintain current duty cycle for 1 second

        # Cleanup
        pwm.disable()
        print("  PWM disabled")
        print("PWM Test completed successfully")

    except Exception as e:
        print(f"PWM Test failed: {e}")

    finally:
        if pwm is not None:
            pwm.close()
            print("  PWM closed")
        if gate is not None:
            gate.write(False)
            gate.close()
            print("MOSFET gate disabled and closed")


if __name__ == "__main__":
    print("Starting PWM Signal Test...\n")
    pwm_test()

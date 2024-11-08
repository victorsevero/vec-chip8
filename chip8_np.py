import argparse
from functools import wraps

import numpy as np
import pygame

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 64, 32
SCALE = 10
MEMORY_SIZE = 4096
REGISTER_COUNT = 16
STACK_SIZE = 16
KEY_COUNT = 16
PROGRAM_START = 0x200

# Font Set for CHIP-8 (each character is 5 bytes, representing 4x5 pixels)
# fmt: off
FONT_SET = np.array(
    [
        0xF0, 0x90, 0x90, 0x90, 0xF0,  # 0
        0x20, 0x60, 0x20, 0x20, 0x70,  # 1
        0xF0, 0x10, 0xF0, 0x80, 0xF0,  # 2
        0xF0, 0x10, 0xF0, 0x10, 0xF0,  # 3
        0x90, 0x90, 0xF0, 0x10, 0x10,  # 4
        0xF0, 0x80, 0xF0, 0x10, 0xF0,  # 5
        0xF0, 0x80, 0xF0, 0x90, 0xF0,  # 6
        0xF0, 0x10, 0x20, 0x40, 0x40,  # 7
        0xF0, 0x90, 0xF0, 0x90, 0xF0,  # 8
        0xF0, 0x90, 0xF0, 0x10, 0xF0,  # 9
        0xF0, 0x90, 0xF0, 0x90, 0x90,  # A
        0xE0, 0x90, 0xE0, 0x90, 0xE0,  # B
        0xF0, 0x80, 0x80, 0x80, 0xF0,  # C
        0xE0, 0x90, 0x90, 0x90, 0xE0,  # D
        0xF0, 0x80, 0xF0, 0x80, 0xF0,  # E
        0xF0, 0x80, 0xF0, 0x80, 0x80   # F
    ],
    dtype=np.uint8,
)
# fmt: on

# CHIP-8 key mapping to keyboard keys
KEY_MAP = {
    pygame.K_1: 0x1,
    pygame.K_2: 0x2,
    pygame.K_3: 0x3,
    pygame.K_4: 0xC,
    pygame.K_q: 0x4,
    pygame.K_w: 0x5,
    pygame.K_e: 0x6,
    pygame.K_r: 0xD,
    pygame.K_a: 0x7,
    pygame.K_s: 0x8,
    pygame.K_d: 0x9,
    pygame.K_f: 0xE,
    pygame.K_z: 0xA,
    pygame.K_x: 0x0,
    pygame.K_c: 0xB,
    pygame.K_v: 0xF,
}


def suppress_overflow_warning(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with np.errstate(over="ignore"):
            return func(*args, **kwargs)

    return wrapper


class Chip8:
    def __init__(self, seed=None):
        self.memory = np.zeros(MEMORY_SIZE, dtype=np.uint8)
        self.memory[: len(FONT_SET)] = FONT_SET
        self.v = np.zeros(REGISTER_COUNT, dtype=np.uint8)  # V0 to VF
        self.i = np.uint16(0)  # Index register
        self.pc = np.uint16(PROGRAM_START)  # Program counter
        self.stack = np.zeros(STACK_SIZE, dtype=np.uint16)
        self.sp = np.uint16(0)  # Stack pointer
        self.delay_timer = np.uint8(0)
        self.sound_timer = np.uint8(0)
        self.display = np.zeros(SCREEN_WIDTH * SCREEN_HEIGHT, dtype=np.uint8)

        # this extra index exists just so that we can evaluate
        # `self.keys[self.pressed_key]` without any issues since numpy doesn't
        # short-circuit bitwise operations and we need to set
        # `self.pressed_key` to `KEY_COUNT` when no key is pressed
        self.keys = np.zeros(KEY_COUNT + 1, dtype=np.uint8)
        self.pressed_key = np.uint8(KEY_COUNT)

        self.rng = np.random.default_rng(seed)

    def load_program(self, program: np.ndarray[np.uint8]):
        self.memory[PROGRAM_START : PROGRAM_START + len(program)] = program

    def fetch_opcode(self) -> np.uint16:
        return self.memory.view(np.uint16)[self.pc // 2].byteswap()

    @suppress_overflow_warning
    def execute_opcode(self, opcode):
        # Extracting nibbles from opcode
        nnn = opcode & 0x0FFF
        n = np.uint8(opcode & 0x000F)
        x = (opcode & 0x0F00) >> 8
        y = (opcode & 0x00F0) >> 4
        kk = np.uint8(opcode)

        # Decode and execute opcode
        if opcode == 0x00E0:
            # Clear the display
            self.display = np.zeros(
                SCREEN_WIDTH * SCREEN_HEIGHT,
                dtype=np.uint8,
            )
        elif opcode == 0x00EE:
            # Return from subroutine
            self.sp -= 1
            self.pc = self.stack[self.sp]
        elif (opcode & 0xF000) == 0x1000:
            # Jump to address NNN
            self.pc = nnn - 2  # Compensate for the increment at the end
        elif (opcode & 0xF000) == 0x2000:
            # Call subroutine at NNN
            self.stack[self.sp] = self.pc
            self.sp += 1
            self.pc = nnn - 2  # Compensate for the increment at the end
        elif (opcode & 0xF000) == 0x3000:
            # Skip next instruction if Vx == kk
            self.pc += np.uint16(np.where(self.v[x] == kk, 2, 0))
        elif (opcode & 0xF000) == 0x4000:
            # Skip next instruction if Vx != kk
            self.pc += np.uint16(np.where(self.v[x] != kk, 2, 0))
        elif (opcode & 0xF000) == 0x5000:
            # Skip next instruction if Vx == Vy
            self.pc += np.uint16(np.where(self.v[x] == self.v[y], 2, 0))
        elif (opcode & 0xF000) == 0x6000:
            # Set Vx = kk
            self.v[x] = kk
        elif (opcode & 0xF000) == 0x7000:
            # Set Vx = Vx + kk
            self.v[x] = self.v[x] + kk
        elif (opcode & 0xF00F) == 0x8000:
            # Set Vx = Vy
            self.v[x] = self.v[y]
        elif (opcode & 0xF00F) == 0x8001:
            # Set Vx = Vx OR Vy
            self.v[x] |= self.v[y]
            self.v[0xF] = 0
        elif (opcode & 0xF00F) == 0x8002:
            # Set Vx = Vx AND Vy
            self.v[x] &= self.v[y]
            self.v[0xF] = 0
        elif (opcode & 0xF00F) == 0x8003:
            # Set Vx = Vx XOR Vy
            self.v[x] ^= self.v[y]
            self.v[0xF] = 0
        elif (opcode & 0xF00F) == 0x8004:
            # Set Vx = Vx + Vy, set VF = carry
            sum_value = np.uint16(self.v[x]) + self.v[y]
            carry = np.uint8(np.where(sum_value > 0xFF, 1, 0))
            self.v[x] = np.uint8(sum_value)
            self.v[0xF] = carry
        elif (opcode & 0xF00F) == 0x8005:
            # Set Vx = Vx - Vy, set VF = NOT borrow
            not_borrow = np.uint8(np.where(self.v[x] >= self.v[y], 1, 0))
            self.v[x] -= self.v[y]
            self.v[0xF] = not_borrow
        elif (opcode & 0xF00F) == 0x8006:
            # Set Vx = Vx SHR 1
            carry = self.v[y] & 0x1
            self.v[x] = self.v[y] >> 1
            self.v[0xF] = carry
        elif (opcode & 0xF00F) == 0x8007:
            # Set Vx = Vy - Vx, set VF = NOT borrow
            not_borrow = np.uint8(np.where(self.v[y] >= self.v[x], 1, 0))
            self.v[x] = self.v[y] - self.v[x]
            self.v[0xF] = not_borrow
        elif (opcode & 0xF00F) == 0x800E:
            # Set Vx = Vx SHL 1
            carry = (self.v[y] & 0x80) >> 7
            self.v[x] = self.v[y] << 1
            self.v[0xF] = carry
        elif (opcode & 0xF00F) == 0x9000:
            # Skip next instruction if Vx != Vy
            self.pc += np.uint16(np.where(self.v[x] != self.v[y], 2, 0))
        elif (opcode & 0xF000) == 0xA000:
            # Set I = nnn
            self.i = nnn
        elif (opcode & 0xF000) == 0xB000:
            # Jump to address NNN + V0
            self.pc = nnn + self.v[0] - 2
            # Compensate for the increment at the end
        elif (opcode & 0xF000) == 0xC000:
            # Set Vx = random byte AND kk
            self.v[x] = self.rng.integers(0, 255, dtype=np.uint8) & kk
        elif (opcode & 0xF000) == 0xD000:
            # Draw sprite at (Vx, Vy) with width 8 pixels and height n
            vx = self.v[x] % SCREEN_WIDTH
            vy = self.v[y] % SCREEN_HEIGHT
            self.v[0xF] = 0

            clipped_x_size = np.uint8(
                np.where(vx + 8 > SCREEN_WIDTH, SCREEN_WIDTH - vx, 8)
            )
            clipped_y_size = np.uint8(
                np.where(vy + n > SCREEN_HEIGHT, SCREEN_HEIGHT - vy, n)
            )

            sprite_bytes = self.memory[self.i : self.i + clipped_y_size]
            flat_sprite = np.unpackbits(sprite_bytes)
            sprite = flat_sprite.reshape(clipped_y_size, 8)[:, :clipped_x_size]
            flat_sprite = sprite.ravel()

            y_indices = vy + np.arange(clipped_y_size)
            x_indices = vx + np.arange(clipped_x_size)

            grid_y, grid_x = np.meshgrid(y_indices, x_indices, indexing="ij")
            flat_indices = (grid_y * SCREEN_WIDTH + grid_x).ravel()

            collision = (self.display[flat_indices] & flat_sprite).any()
            self.v[0xF] = np.uint8(collision)

            self.display[flat_indices] ^= flat_sprite

        elif (opcode & 0xF0FF) == 0xE09E:
            # Skip next instruction if key with the value of Vx is pressed
            self.pc += np.uint16(np.where(self.keys[self.v[x]] == 1, 2, 0))
        elif (opcode & 0xF0FF) == 0xE0A1:
            # Skip next instruction if key with the value of Vx is not pressed
            self.pc += np.uint16(np.where(self.keys[self.v[x]] == 0, 2, 0))
        elif (opcode & 0xF0FF) == 0xF007:
            # Set Vx = delay timer value
            self.v[x] = self.delay_timer
        elif (opcode & 0xF0FF) == 0xF00A:
            just_pressed = (self.pressed_key == KEY_COUNT) & np.any(
                self.keys[:-1]
            )
            just_released = (self.pressed_key != KEY_COUNT) & (
                self.keys[self.pressed_key] == 0
            )
            self.pressed_key = np.uint8(
                np.where(just_pressed, np.argmax(self.keys), self.pressed_key)
            )

            # We should compensate for the increment at the end if a key was
            # not recently released (that is we should keep inside this "loop")
            self.pc -= np.uint16(np.where(just_released, 0, 2))

            self.v[x] = np.uint8(
                np.where(just_released, self.pressed_key, self.v[x])
            )
            self.pressed_key = np.uint8(
                np.where(
                    just_released,
                    KEY_COUNT,
                    self.pressed_key,
                )
            )

        elif (opcode & 0xF0FF) == 0xF015:
            # Set delay timer = Vx
            self.delay_timer = self.v[x]
        elif (opcode & 0xF0FF) == 0xF018:
            # Set sound timer = Vx
            self.sound_timer = self.v[x]
        elif (opcode & 0xF0FF) == 0xF01E:
            # Set I = I + Vx
            self.i += self.v[x]
        elif (opcode & 0xF0FF) == 0xF029:
            # Set I = location of sprite for digit Vx
            self.i = self.v[x] * 5
        elif (opcode & 0xF0FF) == 0xF033:
            # Store BCD representation of Vx in memory locations I, I+1, and I+2
            self.memory[self.i] = self.v[x] // 100
            self.memory[self.i + 1] = (self.v[x] // 10) % 10
            self.memory[self.i + 2] = self.v[x] % 10
        elif (opcode & 0xF0FF) == 0xF055:
            # Store registers V0 through Vx in memory starting at location I
            self.memory[self.i : self.i + x + 1] = self.v[: x + 1]
            self.i += x + 1
        elif (opcode & 0xF0FF) == 0xF065:
            # Read registers V0 through Vx from memory starting at location I
            self.v[: x + 1] = self.memory[self.i : self.i + x + 1]
            self.i += x + 1
        else:
            print(f"Unknown opcode: {opcode:04X}")

        # Move to the next instruction
        self.pc += 2

    @suppress_overflow_warning
    def update_timers(self):
        # Update timers
        self.delay_timer = np.uint8(
            np.where(self.delay_timer > 0, self.delay_timer - 1, 0)
        )
        self.sound_timer = np.uint8(
            np.where(self.sound_timer > 0, self.sound_timer - 1, 0)
        )

    def cycle(self):
        # Fetch, decode, and execute
        opcode = self.fetch_opcode()
        self.execute_opcode(opcode)


def main(game_filename):
    # Initialize emulator and graphics
    chip8 = Chip8(seed=666)
    pygame.init()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH * SCALE, SCREEN_HEIGHT * SCALE)
    )
    pygame.display.set_caption("CHIP-8 Interpreter")

    # Load a program from game file
    with open(game_filename, "rb") as f:
        program = f.read()
    chip8.load_program(np.frombuffer(program, dtype=np.uint8))

    # always CHIP-8 for quirks test ROM
    if "5_quirks" in game_filename:
        chip8.memory[0x1FF] = 1

    clock = pygame.time.Clock()

    # Main loop
    running = True
    n_cycles = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_MAP:
                    chip8.keys[KEY_MAP[event.key]] = 1
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    chip8.keys[KEY_MAP[event.key]] = 0

        chip8.cycle()

        # assuming cycle is 500Hz (2ms) and timers are 60Hz (16.7ms)
        if n_cycles % 8 == 0:
            chip8.update_timers()
            # Draw display
            screen.fill((0, 0, 0))
            display_pixels = chip8.display.copy().reshape(
                (SCREEN_HEIGHT, SCREEN_WIDTH)
            )
            display_pixels *= 255
            display_pixels = np.repeat(display_pixels, SCALE, axis=0)
            display_pixels = np.repeat(display_pixels, SCALE, axis=1)

            surface = pygame.surfarray.make_surface(display_pixels.T)
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick()
            print("FPS:", int(clock.get_fps()), end="\r")

        n_cycles += 1

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CHIP-8 Interpreter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("game", help="Game Filename without extension")
    args = parser.parse_args()
    config = vars(args)
    main(f"games/{config['game']}.ch8")

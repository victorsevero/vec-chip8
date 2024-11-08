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
    def __init__(self, n_emulators, seed=None):
        self.memory = np.zeros((n_emulators, MEMORY_SIZE), dtype=np.uint8)
        self.memory[:, : len(FONT_SET)] = FONT_SET
        # V0 to VF
        self.v = np.zeros((n_emulators, REGISTER_COUNT), dtype=np.uint8)
        # Index register
        self.i = np.zeros(n_emulators, dtype=np.uint16)
        # Program counter
        self.pc = np.full(n_emulators, PROGRAM_START, dtype=np.uint16)
        self.stack = np.zeros((n_emulators, STACK_SIZE), dtype=np.uint16)
        # Stack pointer
        self.sp = np.zeros(n_emulators, dtype=np.uint16)
        self.delay_timer = np.zeros(n_emulators, dtype=np.uint16)
        self.sound_timer = np.zeros(n_emulators, dtype=np.uint16)
        self.display = np.zeros(
            (n_emulators, SCREEN_WIDTH * SCREEN_HEIGHT),
            dtype=np.uint8,
        )

        # this extra index exists just so that we can evaluate
        # `self.keys[self.pressed_key]` without any issues since numpy doesn't
        # short-circuit bitwise operations and we need to set
        # `self.pressed_key` to `KEY_COUNT` when no key is pressed
        self.keys = np.zeros((n_emulators, KEY_COUNT + 1), dtype=np.uint8)
        self.pressed_key = np.full(n_emulators, KEY_COUNT, dtype=np.uint8)

        self.rng = np.random.default_rng(seed)
        self.n_emulators = n_emulators

    def load_program(self, program: np.ndarray[np.uint8]):
        self.memory[:, PROGRAM_START : PROGRAM_START + len(program)] = program

    def fetch_opcode(self) -> np.ndarray[np.uint16]:
        return self.memory.view(np.uint16)[
            np.arange(self.n_emulators),
            self.pc // 2,
        ].byteswap()

    @suppress_overflow_warning
    def execute_opcode(self, opcode: np.ndarray[np.uint16]):
        # Extracting nibbles from opcode
        nnn = opcode & 0x0FFF
        n = np.uint8(opcode & 0x000F)
        x = (opcode & 0x0F00) >> 8
        y = (opcode & 0x00F0) >> 4
        kk = np.uint8(opcode)

        # TODO: remove elif's since pc might not be the same for all of them
        # also, maybe try checking all operations at once and call them after

        # Decode and execute opcode
        if (opcode == 0x00E0).any():
            opcode_idxs = np.nonzero(opcode == 0x00E0)[0]
            # Clear the display
            self.display[opcode_idxs] = np.zeros(
                SCREEN_WIDTH * SCREEN_HEIGHT,
                dtype=np.uint8,
            )
        if (opcode == 0x00EE).any():
            opcode_idxs = np.nonzero(opcode == 0x00EE)[0]
            # Return from subroutine
            self.sp[opcode_idxs] -= 1
            self.pc[opcode_idxs] = self.stack[
                opcode_idxs,
                self.sp[opcode_idxs],
            ]
        if (opcode & 0xF000 == 0x1000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0x1000)[0]
            # Jump to address NNN
            # Compensate for the increment at the end
            self.pc[opcode_idxs] = nnn[opcode_idxs] - 2
        if (opcode & 0xF000 == 0x2000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0x2000)[0]
            # Call subroutine at NNN
            self.stack[opcode_idxs, self.sp[opcode_idxs]] = self.pc[
                opcode_idxs
            ]
            self.sp[opcode_idxs] += 1
            # Compensate for the increment at the end
            self.pc[opcode_idxs] = nnn[opcode_idxs] - 2
        if (opcode & 0xF000 == 0x3000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0x3000)[0]
            # Skip next instruction if Vx == kk
            self.pc[opcode_idxs] += np.uint16(
                np.where(
                    self.v[opcode_idxs, x[opcode_idxs]] == kk[opcode_idxs],
                    2,
                    0,
                )
            )
        if (opcode & 0xF000 == 0x4000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0x4000)[0]
            # Skip next instruction if Vx != kk
            self.pc[opcode_idxs] += np.uint16(
                np.where(
                    self.v[opcode_idxs, x[opcode_idxs]] != kk[opcode_idxs],
                    2,
                    0,
                )
            )
        if (opcode & 0xF000 == 0x5000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0x5000)[0]
            # Skip next instruction if Vx == Vy
            self.pc[opcode_idxs] += np.uint16(
                np.where(
                    self.v[opcode_idxs, x[opcode_idxs]]
                    == self.v[opcode_idxs, y[opcode_idxs]],
                    2,
                    0,
                )
            )
        if (opcode & 0xF000 == 0x6000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0x6000)[0]
            # Set Vx = kk
            self.v[opcode_idxs, x[opcode_idxs]] = kk[opcode_idxs]
        if (opcode & 0xF000 == 0x7000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0x7000)[0]
            # Set Vx = Vx + kk
            self.v[opcode_idxs, x[opcode_idxs]] += kk[opcode_idxs]
        if (opcode & 0xF00F == 0x8000).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x8000)[0]
            # Set Vx = Vy
            self.v[opcode_idxs, x[opcode_idxs]] = self.v[
                opcode_idxs,
                y[opcode_idxs],
            ]
        if (opcode & 0xF00F == 0x8001).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x8001)[0]
            # Set Vx = Vx OR Vy
            self.v[opcode_idxs, x[opcode_idxs]] |= self.v[
                opcode_idxs,
                y[opcode_idxs],
            ]
            self.v[opcode_idxs, 0xF] = 0
        if (opcode & 0xF00F == 0x8002).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x8002)[0]
            # Set Vx = Vx AND Vy
            self.v[opcode_idxs, x[opcode_idxs]] &= self.v[
                opcode_idxs,
                y[opcode_idxs],
            ]
            self.v[opcode_idxs, 0xF] = 0
        if (opcode & 0xF00F == 0x8003).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x8003)[0]
            # Set Vx = Vx XOR Vy
            self.v[opcode_idxs, x[opcode_idxs]] ^= self.v[
                opcode_idxs,
                y[opcode_idxs],
            ]
            self.v[opcode_idxs, 0xF] = 0
        if (opcode & 0xF00F == 0x8004).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x8004)[0]
            # Set Vx = Vx + Vy, set VF = carry
            sum_value = (
                np.uint16(self.v[opcode_idxs, x[opcode_idxs]])
                + self.v[opcode_idxs, y[opcode_idxs]]
            )
            carry = np.uint8(np.where(sum_value > 0xFF, 1, 0))
            self.v[opcode_idxs, x[opcode_idxs]] = np.uint8(sum_value)
            self.v[opcode_idxs, 0xF] = carry
        if (opcode & 0xF00F == 0x8005).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x8005)[0]
            # Set Vx = Vx - Vy, set VF = NOT borrow
            not_borrow = np.uint8(
                np.where(
                    self.v[opcode_idxs, x[opcode_idxs]]
                    >= self.v[opcode_idxs, y[opcode_idxs]],
                    1,
                    0,
                )
            )
            self.v[opcode_idxs, x[opcode_idxs]] -= self.v[
                opcode_idxs,
                y[opcode_idxs],
            ]
            self.v[opcode_idxs, 0xF] = not_borrow
        if (opcode & 0xF00F == 0x8006).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x8006)[0]
            # Set Vx = Vx SHR 1
            carry = self.v[opcode_idxs, y[opcode_idxs]] & 0x1
            self.v[opcode_idxs, x[opcode_idxs]] = (
                self.v[opcode_idxs, y[opcode_idxs]] >> 1
            )
            self.v[opcode_idxs, 0xF] = carry
        if (opcode & 0xF00F == 0x8007).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x8007)[0]
            # Set Vx = Vy - Vx, set VF = NOT borrow
            not_borrow = np.uint8(
                np.where(
                    self.v[opcode_idxs, y[opcode_idxs]]
                    >= self.v[opcode_idxs, x[opcode_idxs]],
                    1,
                    0,
                )
            )
            self.v[opcode_idxs, x[opcode_idxs]] = (
                self.v[opcode_idxs, y[opcode_idxs]]
                - self.v[opcode_idxs, x[opcode_idxs]]
            )
            self.v[opcode_idxs, 0xF] = not_borrow
        if (opcode & 0xF00F == 0x800E).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x800E)[0]
            # Set Vx = Vx SHL 1
            carry = (self.v[opcode_idxs, y[opcode_idxs]] & 0x80) >> 7
            self.v[opcode_idxs, x[opcode_idxs]] = (
                self.v[opcode_idxs, y[opcode_idxs]] << 1
            )
            self.v[opcode_idxs, 0xF] = carry
        if (opcode & 0xF00F == 0x9000).any():
            opcode_idxs = np.nonzero(opcode & 0xF00F == 0x9000)[0]
            # Skip next instruction if Vx != Vy
            self.pc[opcode_idxs] += np.uint16(
                np.where(
                    self.v[opcode_idxs, x[opcode_idxs]]
                    != self.v[opcode_idxs, y[opcode_idxs]],
                    2,
                    0,
                )
            )
        if (opcode & 0xF000 == 0xA000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0xA000)[0]
            # Set I = nnn
            self.i[opcode_idxs] = nnn[opcode_idxs]
        if (opcode & 0xF000 == 0xB000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0xB000)[0]
            # Jump to address NNN + V0
            self.pc[opcode_idxs] = (
                nnn[opcode_idxs] + self.v[opcode_idxs, 0] - 2
            )
            # Compensate for the increment at the end
        if (opcode & 0xF000 == 0xC000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0xC000)[0]
            # Set Vx = random byte AND kk
            self.v[opcode_idxs, x[opcode_idxs]] = (
                self.rng.integers(
                    0,
                    255,
                    size=opcode_idxs.size,
                    dtype=np.uint8,
                )
                & kk[opcode_idxs]
            )
        if (opcode & 0xF000 == 0xD000).any():
            opcode_idxs = np.nonzero(opcode & 0xF000 == 0xD000)[0]
            n_ems = opcode_idxs.size
            # Draw sprite at (Vx, Vy) with width 8 pixels and height n
            vx = self.v[opcode_idxs, x[opcode_idxs]] % SCREEN_WIDTH
            vy = self.v[opcode_idxs, y[opcode_idxs]] % SCREEN_HEIGHT
            self.v[opcode_idxs, 0xF] = 0

            clipped_x_size = np.minimum(SCREEN_WIDTH - vx, [8] * n_ems)
            clipped_y_size = np.minimum(SCREEN_HEIGHT - vy, n[opcode_idxs])

            max_size_y = clipped_y_size.max()
            base_range_y = np.arange(max_size_y)
            ranges_y = np.repeat(base_range_y[np.newaxis, :], n_ems, axis=0)
            mask_y = (
                base_range_y[np.newaxis, :] < clipped_y_size[:, np.newaxis]
            )

            max_size_x = 8
            base_range_x = np.arange(max_size_x)
            ranges_x = np.repeat(base_range_x[np.newaxis, :], n_ems, axis=0)
            mask_x = (
                base_range_x[np.newaxis, :] < clipped_x_size[:, np.newaxis]
            )

            sprite_bytes = self.memory[
                np.repeat(
                    opcode_idxs[:, np.newaxis],
                    max_size_y,
                    axis=1,
                ),
                self.i[opcode_idxs, np.newaxis] + ranges_y,
            ]
            flat_sprite = np.unpackbits(sprite_bytes, axis=1)
            flat_sprite_mask = np.repeat(
                mask_y[:, :, np.newaxis],
                max_size_x,
                axis=2,
            )
            flat_sprite_mask = flat_sprite_mask.swapaxes(0, 1) * mask_x
            # TODO: check if double swapaxes was really necessary
            flat_sprite_mask = flat_sprite_mask.swapaxes(0, 1).reshape(
                n_ems,
                max_size_y * max_size_x,
            )

            y_indices = vy[:, np.newaxis] + ranges_y
            x_indices = vx[:, np.newaxis] + ranges_x

            grid_y, grid_x = np.broadcast_arrays(
                y_indices[:, :, None],
                x_indices[:, None, :],
            )
            # we need to clip the indices to the screen size here. in practice,
            # it won't really matter because indices greater than the screen
            # size will be masked anyway, but we avoid invalid indexing
            indices = (grid_y * SCREEN_WIDTH + grid_x) % 2048
            flat_indices = indices.reshape(n_ems, max_size_y * max_size_x)
            repeat_op_idxs = np.repeat(
                opcode_idxs[:, np.newaxis],
                max_size_y * max_size_x,
                axis=1,
            )
            collision = (
                self.display[repeat_op_idxs, flat_indices]
                & (flat_sprite & flat_sprite_mask)
            ).any(axis=1)
            self.v[opcode_idxs, 0xF] = collision.astype(np.uint8)

            self.display[repeat_op_idxs, flat_indices] ^= (
                flat_sprite & flat_sprite_mask
            )

        if (opcode & 0xF0FF == 0xE09E).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xE09E)[0]
            # Skip next instruction if key with the value of Vx is pressed
            self.pc[opcode_idxs] += np.uint16(
                np.where(
                    self.keys[opcode_idxs, self.v[opcode_idxs, x[opcode_idxs]]]
                    == 1,
                    2,
                    0,
                )
            )
        if (opcode & 0xF0FF == 0xE0A1).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xE0A1)[0]
            # Skip next instruction if key with the value of Vx is not pressed
            self.pc[opcode_idxs] += np.uint16(
                np.where(
                    self.keys[opcode_idxs, self.v[opcode_idxs, x[opcode_idxs]]]
                    == 0,
                    2,
                    0,
                )
            )
        if (opcode & 0xF0FF == 0xF007).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xF007)[0]
            # Set Vx = delay timer value
            self.v[opcode_idxs, x[opcode_idxs]] = self.delay_timer[opcode_idxs]
        if (opcode & 0xF0FF == 0xF00A).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xF00A)[0]
            n_ems = opcode_idxs.size
            # Wait for a key press, store the value of the key in Vx
            just_pressed = (
                self.pressed_key[opcode_idxs] == KEY_COUNT
            ) & np.any(self.keys[opcode_idxs, :-1], axis=1)
            just_released = (self.pressed_key[opcode_idxs] != KEY_COUNT) & (
                self.keys[opcode_idxs, self.pressed_key[opcode_idxs]] == 0
            )
            self.pressed_key[opcode_idxs] = np.uint8(
                np.where(
                    just_pressed,
                    np.argmax(self.keys[opcode_idxs], axis=1),
                    self.pressed_key[opcode_idxs],
                )
            )

            # We should compensate for the increment at the end if a key was
            # not recently released (that is we should keep inside this "loop")
            self.pc[opcode_idxs] -= np.uint16(np.where(just_released, 0, 2))

            self.v[opcode_idxs, x[opcode_idxs]] = np.uint8(
                np.where(
                    just_released,
                    self.pressed_key[opcode_idxs],
                    self.v[opcode_idxs, x[opcode_idxs]],
                )
            )
            self.pressed_key[opcode_idxs] = np.uint8(
                np.where(
                    just_released,
                    np.full(n_ems, KEY_COUNT),
                    self.pressed_key[opcode_idxs],
                )
            )

        if (opcode & 0xF0FF == 0xF015).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xF015)[0]
            # Set delay timer = Vx
            self.delay_timer[opcode_idxs] = self.v[opcode_idxs, x[opcode_idxs]]
        if (opcode & 0xF0FF == 0xF018).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xF018)[0]
            # Set sound timer = Vx
            self.sound_timer[opcode_idxs] = self.v[opcode_idxs, x[opcode_idxs]]
        if (opcode & 0xF0FF == 0xF01E).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xF01E)[0]
            # Set I = I + Vx
            self.i[opcode_idxs] += self.v[opcode_idxs, x[opcode_idxs]]
        if (opcode & 0xF0FF == 0xF029).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xF029)[0]
            # Set I = location of sprite for digit Vx
            self.i[opcode_idxs] = self.v[opcode_idxs, x[opcode_idxs]] * 5
        if (opcode & 0xF0FF == 0xF033).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xF033)[0]
            # Store BCD representation of Vx in memory locations I, I+1, and I+2
            self.memory[opcode_idxs, self.i[opcode_idxs]] = (
                self.v[opcode_idxs, x[opcode_idxs]] // 100
            )
            self.memory[opcode_idxs, self.i[opcode_idxs] + 1] = (
                self.v[opcode_idxs, x[opcode_idxs]] // 10
            ) % 10
            self.memory[opcode_idxs, self.i[opcode_idxs] + 2] = (
                self.v[opcode_idxs, x[opcode_idxs]] % 10
            )
        if (opcode & 0xF0FF == 0xF055).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xF055)[0]
            # Store registers V0 through Vx in memory starting at location I
            max_x = x[opcode_idxs].max()
            range_x = np.arange(max_x + 1)
            mask = range_x < x[opcode_idxs][:, np.newaxis] + 1
            row_indices, col_indices = np.nonzero(mask)

            self.memory[
                opcode_idxs[row_indices],
                self.i[opcode_idxs[row_indices]] + col_indices,
            ] = self.v[opcode_idxs[row_indices], col_indices]

            self.i += x + 1
        if (opcode & 0xF0FF == 0xF065).any():
            opcode_idxs = np.nonzero(opcode & 0xF0FF == 0xF065)[0]
            # Read registers V0 through Vx from memory starting at location I
            max_x = x[opcode_idxs].max()
            range_x = np.arange(max_x + 1)
            mask = range_x < x[opcode_idxs][:, np.newaxis] + 1
            row_indices, col_indices = np.nonzero(mask)

            self.v[opcode_idxs[row_indices], col_indices] = self.memory[
                opcode_idxs[row_indices],
                self.i[opcode_idxs[row_indices]] + col_indices,
            ]
            self.i[opcode_idxs] += x[opcode_idxs] + 1

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


def find_best_grid(n_emulators: int) -> tuple[int, int]:
    """Find best grid for n_emulators based on a widescreen aspect ratio

    Parameters
    ----------
    n_emulators : int
        Number of emulators to fit in the grid

    Returns
    -------
    (int, int)
        Best grid dimensions
    """
    target_ratio = 16 / 9
    best_m, best_n = 1, n_emulators
    best_diff = float("inf")

    for n in range(1, n_emulators + 1):
        m = int(np.ceil(n_emulators / n))

        width = SCREEN_WIDTH * n
        height = SCREEN_HEIGHT * m
        aspect_ratio = width / height

        diff = abs(aspect_ratio - target_ratio)

        if diff < best_diff:
            best_diff = diff
            best_m, best_n = m, n

    return best_m, best_n


def main(game_filename, n_emulators):
    # Initialize emulator and graphics
    chip8 = Chip8(n_emulators=n_emulators, seed=666)
    m, n = find_best_grid(n_emulators)
    pygame.init()
    screen = pygame.display.set_mode(
        (n * SCREEN_WIDTH * SCALE, m * SCREEN_HEIGHT * SCALE)
    )
    pygame.display.set_caption("CHIP-8 Interpreter")

    # Load a program from game file
    with open(game_filename, "rb") as f:
        program = f.read()
    chip8.load_program(np.frombuffer(program, dtype=np.uint8))

    # always CHIP-8 for quirks test ROM
    if "5_quirks" in game_filename:
        chip8.memory[:, 0x1FF] = 1

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
                    chip8.keys[np.arange(n_emulators), KEY_MAP[event.key]] = 1
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    chip8.keys[np.arange(n_emulators), KEY_MAP[event.key]] = 0

        chip8.cycle()

        if n_cycles % 8 == 0:
            chip8.update_timers()
            # Draw display
            screen.fill((0, 0, 0))
            filled_display = np.concatenate(
                (
                    chip8.display,
                    np.zeros(
                        (m * n - n_emulators, SCREEN_WIDTH * SCREEN_HEIGHT),
                        dtype=np.uint8,
                    ),
                )
            )
            display_pixels = (
                filled_display.reshape((m, n, SCREEN_HEIGHT, SCREEN_WIDTH))
                .swapaxes(1, 2)
                .reshape((m * SCREEN_HEIGHT, n * SCREEN_WIDTH))
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
        description="Vectorized CHIP-8 Interpreter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("game", help="Game Filename without extension")
    # a nice -n to display is 16*9*2^(2N+1), because width is double the height
    parser.add_argument("--n_emulators", "-n", type=int, default=1152)
    parser.add_argument("--scale", "-s", type=int, default=1)
    args = parser.parse_args()
    config = vars(args)
    SCALE = config["scale"]
    main(f"games/{config['game']}.ch8", n_emulators=config["n_emulators"])

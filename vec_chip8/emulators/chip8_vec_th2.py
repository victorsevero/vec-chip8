import argparse
from time import perf_counter

import numpy as np
import pygame
import torch as th

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
FONT_SET = th.tensor(
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
    dtype=th.uint8,
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


class Chip8:
    def __init__(self, n_emulators, seed=None, device=None):
        if device is None:
            self.device = th.device(
                "cuda" if th.cuda.is_available() else "cpu"
            )
        else:
            self.device = device
        self.memory = th.zeros(
            (n_emulators, MEMORY_SIZE),
            dtype=th.uint8,
            device=self.device,
        )
        self.memory[:, : len(FONT_SET)] = FONT_SET
        # V0 to VF
        self.v = th.zeros(
            (n_emulators, REGISTER_COUNT),
            dtype=th.uint8,
            device=self.device,
        )
        # Index register
        self.i = th.zeros(n_emulators, dtype=th.int, device=self.device)
        # Program counter
        self.pc = th.full(
            (n_emulators,),
            PROGRAM_START,
            dtype=th.int,
            device=self.device,
        )
        self.stack = th.zeros(
            (n_emulators, STACK_SIZE),
            dtype=th.int,
            device=self.device,
        )
        # Stack pointer
        self.sp = th.zeros(n_emulators, dtype=th.int, device=self.device)
        self.delay_timer = th.zeros(
            n_emulators,
            dtype=th.int,
            device=self.device,
        )
        self.display = th.zeros(
            (n_emulators, SCREEN_WIDTH * SCREEN_HEIGHT),
            dtype=th.uint8,
            device=self.device,
        )

        # this extra index exists just so that we can evaluate
        # `self.keys[self.pressed_key]` without any issues since numpy doesn't
        # short-circuit bitwise operations and we need to set
        # `self.pressed_key` to `KEY_COUNT` when no key is pressed
        self.keys = th.zeros(
            (n_emulators, KEY_COUNT + 1),
            dtype=th.uint8,
            device=self.device,
        )
        self.pressed_key = th.full(
            (n_emulators,),
            KEY_COUNT,
            dtype=th.uint8,
            device=self.device,
        )

        self.rng = th.Generator(device=self.device)
        self.rng.manual_seed(seed)
        self.n_emulators = n_emulators

        self.ZEROS = th.zeros(
            (self.n_emulators,),
            dtype=th.int,
            device=self.device,
        )
        self.opcode_methods = {
            0x00E0: self.op_00E0,
            0x00EE: self.op_00EE,
            0x1000: self.op_1nnn,
            0x2000: self.op_2nnn,
            0x3000: self.op_3xkk,
            0x4000: self.op_4xkk,
            0x5000: self.op_5xy0,
            0x6000: self.op_6xkk,
            0x7000: self.op_7xkk,
            0x8000: self.op_8xy0,
            0x8001: self.op_8xy1,
            0x8002: self.op_8xy2,
            0x8003: self.op_8xy3,
            0x8004: self.op_8xy4,
            0x8005: self.op_8xy5,
            0x8006: self.op_8xy6,
            0x8007: self.op_8xy7,
            0x800E: self.op_8xyE,
            0x9000: self.op_9xy0,
            0xA000: self.op_Annn,
            0xB000: self.op_Bnnn,
            0xC000: self.op_Cxkk,
            0xD000: self.op_Dxyn,
            0xE09E: self.op_Ex9E,
            0xE0A1: self.op_ExA1,
            0xF007: self.op_Fx07,
            0xF00A: self.op_Fx0A,
            0xF015: self.op_Fx15,
            0xF01E: self.op_Fx1E,
            0xF029: self.op_Fx29,
            0xF033: self.op_Fx33,
            0xF055: self.op_Fx55,
            0xF065: self.op_Fx65,
        }
        self.opcode_formats = np.array(list(self.opcode_methods.keys()))

    def load_program(self, program: th.ByteTensor):
        self.memory[:, PROGRAM_START : PROGRAM_START + len(program)] = program

    def fetch_opcode(self) -> th.IntTensor:
        preswap_tensor = self.memory.view(th.int16)[
            th.arange(self.n_emulators, dtype=th.int32, device=self.device),
            self.pc // 2,
        ]
        return (
            preswap_tensor.view(th.uint8)
            .view(-1, 2)[:, [1, 0]]
            .view(th.uint16)
            .reshape_as(preswap_tensor)
            .int()
        )

    def execute_opcode(self, opcode: th.IntTensor):
        # Extracting nibbles from opcode
        self.nnn = opcode & 0x0FFF
        self.n = opcode & 0x000F
        self.x = (opcode & 0x0F00) >> 8
        self.y = (opcode & 0x00F0) >> 4
        self.kk = opcode.type(th.uint8)

        opcode_formats = (
            th.vstack(
                [opcode, opcode & 0xF000, opcode & 0xF00F, opcode & 0xF0FF]
            )
            .cpu()
            .numpy()
        )
        exec_opcodes = np.unique(
            opcode_formats[np.isin(opcode_formats, self.opcode_formats)]
        )
        for exec_opcode in exec_opcodes:
            self.opcode_methods[exec_opcode](opcode)

        # Move to the next instruction
        self.pc += 2

    def op_00E0(self, opcode):
        opcode_idxs = th.nonzero(opcode == 0x00E0).squeeze(1)
        # Clear the display
        self.display[opcode_idxs] = th.zeros(
            SCREEN_WIDTH * SCREEN_HEIGHT,
            dtype=th.uint8,
            device=self.device,
        )

    def op_00EE(self, opcode):
        opcode_idxs = th.nonzero(opcode == 0x00EE).squeeze(1)
        # Return from subroutine
        self.sp[opcode_idxs] -= 1
        self.pc[opcode_idxs] = self.stack[
            opcode_idxs,
            self.sp[opcode_idxs],
        ]

    def op_1nnn(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0x1000).squeeze(1)
        # Jump to address NNN
        # Compensate for the increment at the end
        self.pc[opcode_idxs] = self.nnn[opcode_idxs] - 2

    def op_2nnn(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0x2000).squeeze(1)
        # Call subroutine at NNN
        self.stack[opcode_idxs, self.sp[opcode_idxs]] = self.pc[opcode_idxs]
        self.sp[opcode_idxs] += 1
        # Compensate for the increment at the end
        self.pc[opcode_idxs] = self.nnn[opcode_idxs] - 2

    def op_3xkk(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0x3000).squeeze(1)
        # Skip next instruction if Vx == kk
        self.pc[opcode_idxs] += th.where(
            self.v[opcode_idxs, self.x[opcode_idxs]] == self.kk[opcode_idxs],
            2,
            0,
        )

    def op_4xkk(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0x4000).squeeze(1)
        # Skip next instruction if Vx != kk
        self.pc[opcode_idxs] += th.where(
            self.v[opcode_idxs, self.x[opcode_idxs]] != self.kk[opcode_idxs],
            2,
            0,
        )

    def op_5xy0(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x5000).squeeze(1)
        # Skip next instruction if Vx == Vy
        self.pc[opcode_idxs] += th.where(
            self.v[opcode_idxs, self.x[opcode_idxs]]
            == self.v[opcode_idxs, self.y[opcode_idxs]],
            2,
            0,
        )

    def op_6xkk(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0x6000).squeeze(1)
        # Set Vx = kk
        self.v[opcode_idxs, self.x[opcode_idxs]] = self.kk[opcode_idxs]

    def op_7xkk(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0x7000).squeeze(1)
        # Set Vx = Vx + kk
        self.v[opcode_idxs, self.x[opcode_idxs]] += self.kk[opcode_idxs]

    def op_8xy0(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x8000).squeeze(1)
        # Set Vx = Vy
        self.v[opcode_idxs, self.x[opcode_idxs]] = self.v[
            opcode_idxs,
            self.y[opcode_idxs],
        ]

    def op_8xy1(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x8001).squeeze(1)
        # Set Vx = Vx OR Vy
        self.v[opcode_idxs, self.x[opcode_idxs]] |= self.v[
            opcode_idxs,
            self.y[opcode_idxs],
        ]
        self.v[opcode_idxs, 0xF] = 0

    def op_8xy2(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x8002).squeeze(1)
        # Set Vx = Vx AND Vy
        self.v[opcode_idxs, self.x[opcode_idxs]] &= self.v[
            opcode_idxs,
            self.y[opcode_idxs],
        ]
        self.v[opcode_idxs, 0xF] = 0

    def op_8xy3(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x8003).squeeze(1)
        # Set Vx = Vx XOR Vy
        self.v[opcode_idxs, self.x[opcode_idxs]] ^= self.v[
            opcode_idxs,
            self.y[opcode_idxs],
        ]
        self.v[opcode_idxs, 0xF] = 0

    def op_8xy4(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x8004).squeeze(1)
        # Set Vx = Vx + Vy, set VF = carry
        sum_value = (
            self.v[opcode_idxs, self.x[opcode_idxs]].int()
            + self.v[opcode_idxs, self.y[opcode_idxs]].int()
        )
        carry = th.where(sum_value > 0xFF, 1, 0).type(th.uint8)
        self.v[opcode_idxs, self.x[opcode_idxs]] = sum_value.type(th.uint8)
        self.v[opcode_idxs, 0xF] = carry

    def op_8xy5(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x8005).squeeze(1)
        # Set Vx = Vx - Vy, set VF = NOT borrow
        not_borrow = th.where(
            self.v[opcode_idxs, self.x[opcode_idxs]]
            >= self.v[opcode_idxs, self.y[opcode_idxs]],
            1,
            0,
        ).type(th.uint8)
        self.v[opcode_idxs, self.x[opcode_idxs]] -= self.v[
            opcode_idxs,
            self.y[opcode_idxs],
        ]
        self.v[opcode_idxs, 0xF] = not_borrow

    def op_8xy6(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x8006).squeeze(1)
        # Set Vx = Vx SHR 1
        carry = self.v[opcode_idxs, self.y[opcode_idxs]] & 0x1
        self.v[opcode_idxs, self.x[opcode_idxs]] = (
            self.v[opcode_idxs, self.y[opcode_idxs]] >> 1
        )
        self.v[opcode_idxs, 0xF] = carry

    def op_8xy7(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x8007).squeeze(1)
        # Set Vx = Vy - Vx, set VF = NOT borrow
        not_borrow = th.where(
            self.v[opcode_idxs, self.y[opcode_idxs]]
            >= self.v[opcode_idxs, self.x[opcode_idxs]],
            1,
            0,
        ).type(th.uint8)
        self.v[opcode_idxs, self.x[opcode_idxs]] = (
            self.v[opcode_idxs, self.y[opcode_idxs]]
            - self.v[opcode_idxs, self.x[opcode_idxs]]
        )
        self.v[opcode_idxs, 0xF] = not_borrow

    def op_8xyE(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x800E).squeeze(1)
        # Set Vx = Vx SHL 1
        carry = (self.v[opcode_idxs, self.y[opcode_idxs]] & 0x80) >> 7
        self.v[opcode_idxs, self.x[opcode_idxs]] = (
            self.v[opcode_idxs, self.y[opcode_idxs]] << 1
        )
        self.v[opcode_idxs, 0xF] = carry

    def op_9xy0(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF00F == 0x9000).squeeze(1)
        # Skip next instruction if Vx != Vy
        self.pc[opcode_idxs] += th.where(
            self.v[opcode_idxs, self.x[opcode_idxs]]
            != self.v[opcode_idxs, self.y[opcode_idxs]],
            2,
            0,
        )

    def op_Annn(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0xA000).squeeze(1)
        # Set I = nnn
        self.i[opcode_idxs] = self.nnn[opcode_idxs]

    def op_Bnnn(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0xB000).squeeze(1)
        # Jump to address NNN + V0
        self.pc[opcode_idxs] = (
            self.nnn[opcode_idxs] + self.v[opcode_idxs, 0] - 2
        )
        # Compensate for the increment at the end

    def op_Cxkk(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0xC000).squeeze(1)
        # Set Vx = random byte AND kk
        self.v[opcode_idxs, self.x[opcode_idxs]] = (
            th.randint(
                0,
                256,
                size=(opcode_idxs.shape[0],),
                generator=self.rng,
                dtype=th.uint8,
                device=self.device,
            )
            & self.kk[opcode_idxs]
        )

    def op_Dxyn(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF000 == 0xD000).squeeze(1)
        n_ems = opcode_idxs.shape[0]
        # Draw sprite at (Vx, Vy) with width 8 pixels and height n
        vx = self.v[opcode_idxs, self.x[opcode_idxs]] % SCREEN_WIDTH
        vy = self.v[opcode_idxs, self.y[opcode_idxs]] % SCREEN_HEIGHT
        self.v[opcode_idxs, 0xF] = 0

        clipped_x_size = th.minimum(
            SCREEN_WIDTH - vx,
            th.IntTensor([8] * n_ems).to(self.device),
        )
        clipped_y_size = th.minimum(SCREEN_HEIGHT - vy, self.n[opcode_idxs])

        max_size_y = clipped_y_size.max()
        base_range_y = th.arange(max_size_y, device=self.device)
        ranges_y = th.repeat_interleave(
            base_range_y[th.newaxis, :],
            n_ems,
            axis=0,
        )
        mask_y = base_range_y[th.newaxis, :] < clipped_y_size[:, th.newaxis]

        max_size_x = 8
        base_range_x = th.arange(max_size_x, device=self.device)
        ranges_x = th.repeat_interleave(
            base_range_x[th.newaxis, :],
            n_ems,
            axis=0,
        )
        mask_x = base_range_x[th.newaxis, :] < clipped_x_size[:, th.newaxis]

        sprite_bytes = self.memory[
            th.repeat_interleave(
                opcode_idxs[:, th.newaxis],
                max_size_y,
                axis=1,
            ),
            self.i[opcode_idxs, th.newaxis] + ranges_y,
        ]
        flat_sprite = th_unpackbits(sprite_bytes)
        flat_sprite_mask = th.repeat_interleave(
            mask_y[:, :, th.newaxis],
            max_size_x,
            axis=2,
        )
        flat_sprite_mask = flat_sprite_mask.swapaxes(0, 1) * mask_x
        # TODO: check if double swapaxes was really necessary
        flat_sprite_mask = flat_sprite_mask.swapaxes(0, 1).reshape(
            n_ems,
            max_size_y * max_size_x,
        )

        y_indices = vy[:, th.newaxis] + ranges_y
        x_indices = vx[:, th.newaxis] + ranges_x

        grid_y, grid_x = th.broadcast_tensors(
            y_indices[:, :, None],
            x_indices[:, None, :],
        )
        # we need to clip the indices to the screen size here. in practice,
        # it won't really matter because indices greater than the screen
        # size will be masked anyway, but we avoid invalid indexing
        indices = (grid_y * SCREEN_WIDTH + grid_x) % 2048
        flat_indices = indices.reshape(n_ems, max_size_y * max_size_x)
        repeat_op_idxs = th.repeat_interleave(
            opcode_idxs[:, th.newaxis],
            max_size_y * max_size_x,
            axis=1,
        )
        collision = (
            self.display[repeat_op_idxs, flat_indices]
            & (flat_sprite & flat_sprite_mask)
        ).any(axis=1)
        self.v[opcode_idxs, 0xF] = collision.type(th.uint8)

        self.display[repeat_op_idxs, flat_indices] ^= (
            flat_sprite & flat_sprite_mask
        )

    def op_Ex9E(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xE09E).squeeze(1)
        # Skip next instruction if key with the value of Vx is pressed
        self.pc[opcode_idxs] += th.where(
            self.keys[
                opcode_idxs,
                self.v[opcode_idxs, self.x[opcode_idxs]].int(),
            ]
            == 1,
            2,
            0,
        )

    def op_ExA1(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xE0A1).squeeze(1)
        # Skip next instruction if key with the value of Vx is not pressed
        self.pc[opcode_idxs] += th.where(
            self.keys[
                opcode_idxs, self.v[opcode_idxs, self.x[opcode_idxs]].int()
            ]
            == 0,
            2,
            0,
        )

    def op_Fx07(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xF007).squeeze(1)
        # Set Vx = delay timer value
        self.v[opcode_idxs, self.x[opcode_idxs]] = self.delay_timer[
            opcode_idxs
        ].type(th.uint8)

    def op_Fx0A(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xF00A).squeeze(1)
        n_ems = opcode_idxs.shape[0]
        # Wait for a key press, store the value of the key in Vx
        just_pressed = (self.pressed_key[opcode_idxs] == KEY_COUNT) & th.any(
            self.keys[opcode_idxs, :-1], axis=1
        )
        just_released = (self.pressed_key[opcode_idxs] != KEY_COUNT) & (
            self.keys[opcode_idxs, self.pressed_key[opcode_idxs].int()] == 0
        )
        self.pressed_key[opcode_idxs] = th.where(
            just_pressed,
            th.argmax(self.keys[opcode_idxs], axis=1),
            self.pressed_key[opcode_idxs],
        ).type(th.uint8)

        # We should compensate for the increment at the end if a key was
        # not recently released (that is we should keep inside this "loop")
        self.pc[opcode_idxs] -= th.where(just_released, 0, 2)

        self.v[opcode_idxs, self.x[opcode_idxs]] = th.where(
            just_released,
            self.pressed_key[opcode_idxs],
            self.v[opcode_idxs, self.x[opcode_idxs]],
        ).type(th.uint8)
        self.pressed_key[opcode_idxs] = th.where(
            just_released,
            th.full((n_ems,), KEY_COUNT, device=self.device),
            self.pressed_key[opcode_idxs],
        ).type(th.uint8)

    def op_Fx15(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xF015).squeeze(1)
        # Set delay timer = Vx
        self.delay_timer[opcode_idxs] = self.v[
            opcode_idxs,
            self.x[opcode_idxs],
        ].int()

    def op_Fx1E(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xF01E).squeeze(1)
        # Set I = I + Vx
        self.i[opcode_idxs] += self.v[opcode_idxs, self.x[opcode_idxs]]

    def op_Fx29(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xF029).squeeze(1)
        # Set I = location of sprite for digit Vx
        self.i[opcode_idxs] = (
            self.v[opcode_idxs, self.x[opcode_idxs]].int() * 5
        )

    def op_Fx33(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xF033).squeeze(1)
        # Store BCD representation of Vx in memory locations I, I+1, and I+2
        self.memory[opcode_idxs, self.i[opcode_idxs]] = (
            self.v[opcode_idxs, self.x[opcode_idxs]] // 100
        )
        self.memory[opcode_idxs, self.i[opcode_idxs] + 1] = (
            self.v[opcode_idxs, self.x[opcode_idxs]] // 10
        ) % 10
        self.memory[opcode_idxs, self.i[opcode_idxs] + 2] = (
            self.v[opcode_idxs, self.x[opcode_idxs]] % 10
        )

    def op_Fx55(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xF055).squeeze(1)
        # Store registers V0 through Vx in memory starting at location I
        max_x = self.x[opcode_idxs].max()
        range_x = th.arange(max_x + 1, device=self.device)
        mask = range_x < self.x[opcode_idxs][:, th.newaxis] + 1
        row_indices, col_indices = th.nonzero(mask, as_tuple=True)

        self.memory[
            opcode_idxs[row_indices],
            self.i[opcode_idxs[row_indices]] + col_indices,
        ] = self.v[opcode_idxs[row_indices], col_indices]

        self.i += self.x + 1

    def op_Fx65(self, opcode):
        opcode_idxs = th.nonzero(opcode & 0xF0FF == 0xF065).squeeze(1)
        # Read registers V0 through Vx from memory starting at location I
        max_x = self.x[opcode_idxs].max()
        range_x = th.arange(max_x + 1, device=self.device)
        mask = range_x < self.x[opcode_idxs][:, th.newaxis] + 1
        row_indices, col_indices = th.nonzero(mask, as_tuple=True)

        self.v[opcode_idxs[row_indices], col_indices] = self.memory[
            opcode_idxs[row_indices],
            self.i[opcode_idxs[row_indices]] + col_indices,
        ]
        self.i[opcode_idxs] += self.x[opcode_idxs] + 1

    def update_timers(self):
        # Update timers
        self.delay_timer = th.maximum(self.delay_timer - 1, self.ZEROS)

    def cycle(self):
        # Fetch, decode, and execute
        opcode = self.fetch_opcode()
        self.execute_opcode(opcode)


def th_unpackbits(tensor: th.Tensor, num_bits=8):
    bits = 2 ** th.arange(
        num_bits - 1,
        -1,
        -1,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    unpacked = (tensor.unsqueeze(-1) & bits).ne(0).type(th.uint8)
    return unpacked.reshape(unpacked.shape[0], -1)


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


def main(game_filename, n_emulators, max_cycles=None, device=None):
    # Initialize emulator and graphics
    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
    else:
        device = th.device(device)
    chip8 = Chip8(n_emulators=n_emulators, seed=666, device=device)
    # m, n = find_best_grid(n_emulators)
    # pygame.init()
    # screen = pygame.display.set_mode(
    #     (n * SCREEN_WIDTH * SCALE, m * SCREEN_HEIGHT * SCALE)
    # )
    # pygame.display.set_caption("CHIP-8 Interpreter")

    # Load a program from game file
    with open(game_filename, "rb") as f:
        program = f.read()
    chip8.load_program(
        th.from_numpy(np.frombuffer(program, dtype=np.uint8).copy())
    )

    # always CHIP-8 for quirks test ROM
    if "5_quirks" in game_filename:
        chip8.memory[:, 0x1FF] = 1

    # clock = pygame.time.Clock()

    # Main loop
    running = True
    n_cycles = 0
    start = perf_counter()
    while running:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key in KEY_MAP:
        #             chip8.keys[
        #                 th.arange(n_emulators, device=device),
        #                 KEY_MAP[event.key],
        #             ] = 1
        #     elif event.type == pygame.KEYUP:
        #         if event.key in KEY_MAP:
        #             chip8.keys[
        #                 th.arange(n_emulators, device=device),
        #                 KEY_MAP[event.key],
        #             ] = 0

        chip8.cycle()

        if n_cycles % 8 == 0:
            chip8.update_timers()
            # Draw display
            # screen.fill((0, 0, 0))
            # filled_display = th.concatenate(
            #     (
            #         chip8.display,
            #         th.zeros(
            #             (m * n - n_emulators, SCREEN_WIDTH * SCREEN_HEIGHT),
            #             dtype=th.uint8,
            #             device=device,
            #         ),
            #     )
            # )
            # display_pixels = (
            #     filled_display.reshape((m, n, SCREEN_HEIGHT, SCREEN_WIDTH))
            #     .swapaxes(1, 2)
            #     .reshape((m * SCREEN_HEIGHT, n * SCREEN_WIDTH))
            # )
            # display_pixels *= 255
            # display_pixels = th.repeat_interleave(
            #     display_pixels,
            #     SCALE,
            #     axis=0,
            # )
            # display_pixels = th.repeat_interleave(
            #     display_pixels,
            #     SCALE,
            #     axis=1,
            # )

            # surface = pygame.surfarray.make_surface(
            #     display_pixels.T.cpu().numpy()
            # )
            # screen.blit(surface, (0, 0))
            # pygame.display.flip()
            # clock.tick()
            # print("FPS:", int(clock.get_fps()), end="\r")

        n_cycles += 1
        if max_cycles is not None and n_cycles >= max_cycles:
            running = False

    end = perf_counter()
    # pygame.quit()

    return end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vectorized CHIP-8 Interpreter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("game", help="Game Filename without extension")
    # a nice -n to display is 16*9*2^(2N+1), because width is double the height
    parser.add_argument("--n_emulators", "-n", type=int, default=2)
    parser.add_argument("--scale", "-s", type=int, default=10)
    parser.add_argument("--max_cycles", "-m", type=int, default=10_000)
    parser.add_argument("--device", "-d", type=str, default=None)
    args = parser.parse_args()
    config = vars(args)
    SCALE = config["scale"]
    main(
        f"games/{config['game']}.ch8",
        n_emulators=config["n_emulators"],
        max_cycles=config["max_cycles"],
        device=config["device"],
    )

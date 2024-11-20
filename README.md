# CHIP-8 Implementation in Python

This project is an attempt to make CHIP-8 interpreters with Python according to the following plan:

- [X] standard implementation
- [X] `numpy` implementation
- [X] vectorized `numpy` implementation
- [X] vectorized `torch` implementation
- [X] vectorized `jax` implementation
- [X] vectorized `cupy` implementation

The objective is to check if implementing vectorized interpreters on CPU and/or GPU is a viable approach and if it offers a performance boost for Reinforcement Learning applications.

# Tests

Test ROMs were taken from [here](https://github.com/Timendus/chip8-test-suite?tab=readme-ov-file#available-tests)

# References

1. http://devernay.free.fr/hacks/chip8/C8TECH10.HTM
2. https://tobiasvl.github.io/blog/write-a-chip-8-emulator
3. https://github.com/Timendus/chip8-test-suite
4. https://github.com/tizian/Chip-8-Emulator/blob/master/Pong%20(1%20player).ch8
5. https://github.com/badlogic/chip8/blob/master/roms/pong.rom

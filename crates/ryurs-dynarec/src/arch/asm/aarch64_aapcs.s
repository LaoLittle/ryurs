.globl __RNvNtNtC13ryurs_dynarec4arch3asm13execute_block
.globl _RNvNtNtC13ryurs_dynarec4arch3asm13execute_block
__RNvNtNtC13ryurs_dynarec4arch3asm13execute_block:
_RNvNtNtC13ryurs_dynarec4arch3asm13execute_block:
    sub sp, sp, #224

    stp x19, x20, [sp, #0]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]

    stp q8, q9, [sp, #96]
    stp q10, q11, [sp, #128]
    stp q12, q13, [sp, #160]
    stp q14, q15, [sp, #192]

    mov x10, sp
    str x10, [x3, #{landing_sp_off}]
    adr x9, 2f
    str x9, [x3, #{landing_pc_off}]
    str x9, [x1, #{return_addr_off}]

    mov x19, x1
    mov x11, x0
    adr x10, 1f
    mov x30, x10
    // The translated block uses Cranelift's `tail` calling convention,
    // which reserves x0/x1 and places user arguments in x2/x3.
    mov x3, x2
    mov x2, x1
    br x11

1:
    str x0, [x19, #{context_pc_off}]
    ldr x9, [x19, #{return_addr_off}]
    br x9

.globl __RNvNtNtC13ryurs_dynarec4arch3asm14landing_target
.globl _RNvNtNtC13ryurs_dynarec4arch3asm14landing_target
__RNvNtNtC13ryurs_dynarec4arch3asm14landing_target:
_RNvNtNtC13ryurs_dynarec4arch3asm14landing_target:
2:
    ldr x0, [x19, #{context_pc_off}]
    ldp q8, q9, [sp, #96]
    ldp q10, q11, [sp, #128]
    ldp q12, q13, [sp, #160]
    ldp q14, q15, [sp, #192]

    ldp x19, x20, [sp, #0]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    ldp x29, x30, [sp, #80]

    add sp, sp, #224
    ret
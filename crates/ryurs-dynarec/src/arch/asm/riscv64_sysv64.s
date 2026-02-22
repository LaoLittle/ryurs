.globl _RNvNtNtC13ryurs_dynarec4arch3asm13execute_block
_RNvNtNtC13ryurs_dynarec4arch3asm13execute_block:
    addi sp, sp, -208

    sd ra, 0(sp)
    sd s0, 8(sp)
    sd s1, 16(sp)
    sd s2, 24(sp)
    sd s3, 32(sp)
    sd s4, 40(sp)
    sd s5, 48(sp)
    sd s6, 56(sp)
    sd s7, 64(sp)
    sd s8, 72(sp)
    sd s9, 80(sp)
    sd s10, 88(sp)
    sd s11, 96(sp)

    fsd fs0, 104(sp)
    fsd fs1, 112(sp)
    fsd fs2, 120(sp)
    fsd fs3, 128(sp)
    fsd fs4, 136(sp)
    fsd fs5, 144(sp)
    fsd fs6, 152(sp)
    fsd fs7, 160(sp)
    fsd fs8, 168(sp)
    fsd fs9, 176(sp)
    fsd fs10, 184(sp)
    fsd fs11, 192(sp)

    mv t1, sp
    sd t1, {landing_sp_off}(a3)

    lla t0, 2f
    sd t0, {landing_pc_off}(a3)
    sd t0, {return_addr_off}(a1)
    mv s11, a1
    lla ra, 1f
    mv t2, a0
    mv a0, a1
    mv a1, a2
    jr t2

1:
    sd a0, {context_pc_off}(s11)
    ld t0, {return_addr_off}(s11)
    jr t0

.globl _RNvNtNtC13ryurs_dynarec4arch3asm14landing_target
_RNvNtNtC13ryurs_dynarec4arch3asm14landing_target:
2:
    ld a0, {context_pc_off}(s11)
    fld fs0, 104(sp)
    fld fs1, 112(sp)
    fld fs2, 120(sp)
    fld fs3, 128(sp)
    fld fs4, 136(sp)
    fld fs5, 144(sp)
    fld fs6, 152(sp)
    fld fs7, 160(sp)
    fld fs8, 168(sp)
    fld fs9, 176(sp)
    fld fs10, 184(sp)
    fld fs11, 192(sp)

    ld s0, 8(sp)
    ld s1, 16(sp)
    ld s2, 24(sp)
    ld s3, 32(sp)
    ld s4, 40(sp)
    ld s5, 48(sp)
    ld s6, 56(sp)
    ld s7, 64(sp)
    ld s8, 72(sp)
    ld s9, 80(sp)
    ld s10, 88(sp)
    ld s11, 96(sp)
    ld ra, 0(sp)

    addi sp, sp, 208
    ret
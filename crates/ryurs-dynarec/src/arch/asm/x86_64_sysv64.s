.globl __RNvNtNtC13ryurs_dynarec4arch3asm13execute_block
.globl _RNvNtNtC13ryurs_dynarec4arch3asm13execute_block
__RNvNtNtC13ryurs_dynarec4arch3asm13execute_block:
_RNvNtNtC13ryurs_dynarec4arch3asm13execute_block:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    push rsi

    mov [rcx + {landing_sp_off}], rsp
    lea r11, [rip + 2f]
    mov [rcx + {landing_pc_off}], r11
    mov [rsi + {return_addr_off}], r11

    mov rax, rdi
    mov rdi, rsi
    mov rsi, rdx
    jmp rax

.globl __RNvNtNtC13ryurs_dynarec4arch3asm14landing_target
.globl _RNvNtNtC13ryurs_dynarec4arch3asm14landing_target
__RNvNtNtC13ryurs_dynarec4arch3asm14landing_target:
_RNvNtNtC13ryurs_dynarec4arch3asm14landing_target:
2:
    mov rcx, [rsp]
    mov rax, [rcx + {context_pc_off}]
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

	.file	"dataCmov.c"
	.text
	.globl	compare
	.type	compare, @function
compare:
.LFB21:
	.cfi_startproc
	cmpq	%rsi, %rdi
	jl	.L4
	movq	%rsi, %rax
	cqto
	idivq	%rdi
	ret
.L4:
	movq	%rdi, %rax
	cqto
	idivq	%rsi
	ret
	.cfi_endproc
.LFE21:
	.size	compare, .-compare
	.globl	ccompare
	.type	ccompare, @function
ccompare:
.LFB22:
	.cfi_startproc
	movq	%rdi, %rax
	cqto
	idivq	%rsi
	movq	%rax, %rcx
	movq	%rsi, %rax
	cqto
	idivq	%rdi
	cmpq	%rcx, %rax
	cmovg	%rcx, %rax
	ret
	.cfi_endproc
.LFE22:
	.size	ccompare, .-ccompare
	.globl	main
	.type	main, @function
main:
.LFB23:
	.cfi_startproc
	movl	$0, %eax
	ret
	.cfi_endproc
.LFE23:
	.size	main, .-main
	.ident	"GCC: (Debian 6.3.0-18+deb9u1) 6.3.0 20170516"
	.section	.note.GNU-stack,"",@progbits

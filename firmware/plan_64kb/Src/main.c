#include "stm32h7xx.h"
#include "nn_engine.h"
#include "nn_model.h"
#include "model_weights.h"
#include <stdio.h>

#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define DWT_LAR    (*(volatile uint32_t *)0xE0001FB0u)

#define CPU_HZ 400000000u
#define CYC_PER_US (CPU_HZ / 1000000u)

static void uart_init(uint32_t bus_hz)
{
    RCC->AHB4ENR |= RCC_AHB4ENR_GPIOBEN;
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
    (void)RCC->APB2ENR;

    GPIOB->MODER &= ~((3u << (14u * 2u)) | (3u << (15u * 2u)));
    GPIOB->MODER |=  ((2u << (14u * 2u)) | (2u << (15u * 2u)));
    GPIOB->AFR[1] &= ~((0xFu << ((14u - 8u) * 4u)) | (0xFu << ((15u - 8u) * 4u)));
    GPIOB->AFR[1] |=  ((4u << ((14u - 8u) * 4u)) | (4u << ((15u - 8u) * 4u)));

    USART1->CR1 = 0u;
    USART1->BRR = bus_hz / 115200u;
    USART1->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

static void uart_putc(char c)
{
    while ((USART1->ISR & USART_ISR_TXE_TXFNF) == 0u) {}
    USART1->TDR = (uint8_t)c;
}

static void uart_wait_done(void)
{
    while ((USART1->ISR & USART_ISR_TC) == 0u) {}
}

int __io_putchar(int ch)
{
    if (ch == '\n') uart_putc('\r');
    uart_putc((char)ch);
    return ch;
}

static void raw_puts(const char *p)
{
    while (*p) { if (*p == '\n') uart_putc('\r'); uart_putc(*p++); }
}

static void raw_hex(uint32_t v)
{
    static const char h[] = "0123456789ABCDEF";
    uart_putc('0'); uart_putc('x');
    for (int i = 28; i >= 0; i -= 4) uart_putc(h[(v >> i) & 0xFu]);
}

void fault_report(uint32_t *sp)
{
    raw_puts("\n\n*** HARD FAULT ***\nPC   = "); raw_hex(sp[6]);
    raw_puts("\nLR   = "); raw_hex(sp[5]);
    raw_puts("\nCFSR = "); raw_hex(SCB->CFSR);
    raw_puts("\nHFSR = "); raw_hex(SCB->HFSR);
    raw_puts("\nMMFAR= "); raw_hex(SCB->MMFAR);
    raw_puts("\nBFAR = "); raw_hex(SCB->BFAR);
    raw_puts("\nSP   = "); raw_hex((uint32_t)sp);
    raw_puts("\n");
    while (1) {}
}

__attribute__((naked)) void HardFault_Handler(void)
{
    __asm volatile(
        "tst lr, #4      \n"
        "ite eq          \n"
        "mrseq r0, msp   \n"
        "mrsne r0, psp   \n"
        "b fault_report  \n");
}

__attribute__((naked)) void MemManage_Handler(void)  { __asm volatile("b HardFault_Handler"); }
__attribute__((naked)) void BusFault_Handler(void)   { __asm volatile("b HardFault_Handler"); }
__attribute__((naked)) void UsageFault_Handler(void) { __asm volatile("b HardFault_Handler"); }

static uint64_t now(void);

static uint64_t now_verbose(void)
{
    static uint32_t calls;
    calls++;
    if ((calls & 1u) == 0u) uart_putc('.');
    return now();
}

static void clock_init(void)
{
    PWR->CR3 = (PWR->CR3 & ~(PWR_CR3_LDOEN | PWR_CR3_BYPASS)) | PWR_CR3_SMPSEN;
    while ((PWR->CSR1 & PWR_CSR1_ACTVOSRDY) == 0u) {}

    PWR->D3CR = (PWR->D3CR & ~PWR_D3CR_VOS) | PWR_D3CR_VOS;
    while ((PWR->D3CR & PWR_D3CR_VOSRDY) == 0u) {}

    FLASH->ACR = FLASH_ACR_LATENCY_2WS | FLASH_ACR_WRHIGHFREQ_1;
    while ((FLASH->ACR & FLASH_ACR_LATENCY) != FLASH_ACR_LATENCY_2WS) {}

    RCC->CR |= RCC_CR_HSEON;
    while ((RCC->CR & RCC_CR_HSERDY) == 0u) {}

    RCC->PLLCKSELR = RCC_PLLCKSELR_PLLSRC_HSE | (5u << RCC_PLLCKSELR_DIVM1_Pos);
    RCC->PLLCFGR = (RCC->PLLCFGR & ~(RCC_PLLCFGR_PLL1RGE | RCC_PLLCFGR_PLL1VCOSEL | RCC_PLLCFGR_PLL1FRACEN))
                 | (2u << RCC_PLLCFGR_PLL1RGE_Pos)
                 | RCC_PLLCFGR_DIVP1EN | RCC_PLLCFGR_DIVQ1EN | RCC_PLLCFGR_DIVR1EN;
    RCC->PLL1DIVR = ((160u - 1u) << RCC_PLL1DIVR_N1_Pos)
                  | ((2u - 1u)   << RCC_PLL1DIVR_P1_Pos)
                  | ((4u - 1u)   << RCC_PLL1DIVR_Q1_Pos)
                  | ((2u - 1u)   << RCC_PLL1DIVR_R1_Pos);
    RCC->CR |= RCC_CR_PLL1ON;
    while ((RCC->CR & RCC_CR_PLL1RDY) == 0u) {}

    uart_wait_done();

    RCC->D1CFGR = RCC_D1CFGR_HPRE_DIV2 | RCC_D1CFGR_D1PPRE_DIV2;
    RCC->D2CFGR = RCC_D2CFGR_D2PPRE1_DIV2 | RCC_D2CFGR_D2PPRE2_DIV2;
    RCC->D3CFGR = RCC_D3CFGR_D3PPRE_DIV2;
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL1;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL1) {}

    uart_init(100000000u);
}

static volatile uint32_t cyc_hi;
static volatile uint32_t cyc_last;

static uint64_t now(void)
{
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    uint32_t c = DWT_CYCCNT;
    if (c < cyc_last) cyc_hi++;
    cyc_last = c;
    uint64_t r = ((uint64_t)cyc_hi << 32) | c;
    __set_PRIMASK(pm);
    return r;
}

void SysTick_Handler(void)
{
    (void)now();
}

static void dwt_init(void)
{
    DEMCR |= (1u << 24);
    DWT_LAR = 0xC5ACCE55u;
    DWT_CYCCNT = 0u;
    cyc_hi = 0u;
    cyc_last = 0u;
    DWT_CTRL |= 1u;

    SysTick->LOAD = 0x00FFFFFFu;
    SysTick->VAL = 0u;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_TICKINT_Msk | SysTick_CTRL_ENABLE_Msk;
}

static void pin_af(GPIO_TypeDef *port, uint32_t pin, uint32_t af, uint32_t pullup)
{
    port->MODER   = (port->MODER   & ~(3u << (pin * 2u))) | (2u << (pin * 2u));
    port->OSPEEDR |= (3u << (pin * 2u));
    port->OTYPER  &= ~(1u << pin);
    port->PUPDR   = (port->PUPDR   & ~(3u << (pin * 2u))) | ((pullup ? 1u : 0u) << (pin * 2u));
    if (pin < 8u) port->AFR[0] = (port->AFR[0] & ~(0xFu << (pin * 4u))) | (af << (pin * 4u));
    else          port->AFR[1] = (port->AFR[1] & ~(0xFu << ((pin - 8u) * 4u))) | (af << ((pin - 8u) * 4u));
}

static void qspi_wait_idle(void)
{
    while (QUADSPI->SR & QUADSPI_SR_BUSY) {}
}

static void qspi_cmd(uint32_t instr, uint32_t imode)
{
    qspi_wait_idle();
    QUADSPI->FCR = QUADSPI_FCR_CTCF;
    QUADSPI->CCR = (imode << QUADSPI_CCR_IMODE_Pos) | (instr << QUADSPI_CCR_INSTRUCTION_Pos);
    while ((QUADSPI->SR & QUADSPI_SR_TCF) == 0u) {}
    QUADSPI->FCR = QUADSPI_FCR_CTCF;
    qspi_wait_idle();
}

static void delay_cycles(uint32_t n)
{
    uint32_t s = DWT_CYCCNT;
    while ((DWT_CYCCNT - s) < n) {}
}

static void qspi_init(void)
{
    RCC->AHB4ENR |= RCC_AHB4ENR_GPIOBEN | RCC_AHB4ENR_GPIOCEN | RCC_AHB4ENR_GPIOFEN |
                    RCC_AHB4ENR_GPIOGEN | RCC_AHB4ENR_GPIOHEN;
    RCC->AHB3ENR |= RCC_AHB3ENR_QSPIEN;
    (void)RCC->AHB3ENR;
    RCC->AHB3RSTR |= RCC_AHB3RSTR_QSPIRST;
    RCC->AHB3RSTR &= ~RCC_AHB3RSTR_QSPIRST;

    pin_af(GPIOB, 2u,  9u,  0u);
    pin_af(GPIOG, 6u,  10u, 1u);
    pin_af(GPIOC, 11u, 9u,  1u);
    pin_af(GPIOF, 8u,  10u, 0u);
    pin_af(GPIOH, 2u,  9u,  0u);
    pin_af(GPIOF, 9u,  10u, 0u);
    pin_af(GPIOH, 3u,  9u,  0u);
    pin_af(GPIOF, 7u,  9u,  0u);
    pin_af(GPIOG, 9u,  9u,  0u);
    pin_af(GPIOF, 6u,  9u,  0u);
    pin_af(GPIOG, 14u, 9u,  0u);

    QUADSPI->CR = 0u;
    QUADSPI->DCR = (26u << QUADSPI_DCR_FSIZE_Pos) | (3u << QUADSPI_DCR_CSHT_Pos);
    QUADSPI->CR = (2u << QUADSPI_CR_PRESCALER_Pos) | (3u << QUADSPI_CR_FTHRES_Pos) |
                  QUADSPI_CR_SSHIFT | QUADSPI_CR_DFM | QUADSPI_CR_EN;

    qspi_cmd(0x66u, 3u);
    qspi_cmd(0x99u, 3u);
    delay_cycles(400000u);
    qspi_cmd(0x66u, 1u);
    qspi_cmd(0x99u, 1u);
    delay_cycles(400000u);

    qspi_wait_idle();
    QUADSPI->CCR = (3u << QUADSPI_CCR_FMODE_Pos) |
                   (1u << QUADSPI_CCR_IMODE_Pos) | (0x6Bu << QUADSPI_CCR_INSTRUCTION_Pos) |
                   (1u << QUADSPI_CCR_ADMODE_Pos) | (2u << QUADSPI_CCR_ADSIZE_Pos) |
                   (3u << QUADSPI_CCR_DMODE_Pos) | (8u << QUADSPI_CCR_DCYC_Pos);
}

#if NN_HAS_SDRAM
static void sdram_wait(void)
{
    while (FMC_Bank5_6_R->SDSR & (1u << 5)) {}
}

static void sdram_cmd(uint32_t mode, uint32_t nrfs, uint32_t mrd)
{
    sdram_wait();
    FMC_Bank5_6_R->SDCMR = (mode << FMC_SDCMR_MODE_Pos) | (1u << FMC_SDCMR_CTB2_Pos) |
                           ((nrfs - 1u) << FMC_SDCMR_NRFS_Pos) | (mrd << FMC_SDCMR_MRD_Pos);
    sdram_wait();
}

static void sdram_init(void)
{
    static const uint8_t pd[] = { 0, 1, 8, 9, 10, 14, 15 };
    static const uint8_t pe[] = { 0, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    static const uint8_t pf[] = { 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15 };
    static const uint8_t pg[] = { 0, 1, 2, 3, 4, 5, 8, 15 };
    static const uint8_t ph[] = { 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    static const uint8_t pi[] = { 0, 1, 2, 3, 4, 5, 6, 7, 9, 10 };

    RCC->AHB4ENR |= RCC_AHB4ENR_GPIODEN | RCC_AHB4ENR_GPIOEEN | RCC_AHB4ENR_GPIOFEN |
                    RCC_AHB4ENR_GPIOGEN | RCC_AHB4ENR_GPIOHEN | RCC_AHB4ENR_GPIOIEN;
    RCC->AHB3ENR |= RCC_AHB3ENR_FMCEN;
    (void)RCC->AHB3ENR;

    for (uint32_t i = 0; i < sizeof(pd); i++) pin_af(GPIOD, pd[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(pe); i++) pin_af(GPIOE, pe[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(pf); i++) pin_af(GPIOF, pf[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(pg); i++) pin_af(GPIOG, pg[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(ph); i++) pin_af(GPIOH, ph[i], 12u, 1u);
    for (uint32_t i = 0; i < sizeof(pi); i++) pin_af(GPIOI, pi[i], 12u, 1u);

    FMC_Bank5_6_R->SDCR[0] = (2u << FMC_SDCRx_SDCLK_Pos) | (1u << FMC_SDCRx_RBURST_Pos) | (0u << FMC_SDCRx_RPIPE_Pos);
    FMC_Bank5_6_R->SDCR[1] = (1u << FMC_SDCRx_NC_Pos) | (1u << FMC_SDCRx_NR_Pos) | (2u << FMC_SDCRx_MWID_Pos) |
                             (1u << FMC_SDCRx_NB_Pos) | (3u << FMC_SDCRx_CAS_Pos);
    FMC_Bank5_6_R->SDTR[0] = ((7u - 1u) << FMC_SDTRx_TRC_Pos) | ((2u - 1u) << FMC_SDTRx_TRP_Pos);
    FMC_Bank5_6_R->SDTR[1] = ((2u - 1u) << FMC_SDTRx_TMRD_Pos) | ((7u - 1u) << FMC_SDTRx_TXSR_Pos) |
                             ((4u - 1u) << FMC_SDTRx_TRAS_Pos) | ((2u - 1u) << FMC_SDTRx_TWR_Pos) |
                             ((2u - 1u) << FMC_SDTRx_TRCD_Pos);
    FMC_Bank1_R->BTCR[0] |= FMC_BCR1_FMCEN_Msk;

    sdram_cmd(1u, 1u, 0u);
    delay_cycles(400000u);
    sdram_cmd(2u, 1u, 0u);
    sdram_cmd(3u, 8u, 0u);
    sdram_cmd(4u, 1u, 0x230u);
    FMC_Bank5_6_R->SDRTR = (0x603u << FMC_SDRTR_COUNT_Pos);
}

static int sdram_test(void)
{
    volatile uint32_t *p = (volatile uint32_t *)0xD0000000u;
    uint32_t n = NN_SDRAM_FLOATS;
    for (uint32_t i = 0; i < n; i++) p[i] = 0xA5A50000u ^ (i * 2654435761u);
    for (uint32_t i = 0; i < n; i++) {
        if (p[i] != (0xA5A50000u ^ (i * 2654435761u))) {
            printf("SDRAM test FAIL at 0x%08lX: read 0x%08lX\n", (unsigned long)(0xD0000000u + i * 4u), (unsigned long)p[i]);
            return 0;
        }
    }
    printf("SDRAM test OK (%lu B at 0xD0000000)\n", (unsigned long)(n * 4u));
    return 1;
}
#endif

static void qspi_abort(void)
{
    QUADSPI->CR |= QUADSPI_CR_ABORT;
    while (QUADSPI->CR & QUADSPI_CR_ABORT) {}
    qspi_wait_idle();
    QUADSPI->FCR = QUADSPI_FCR_CTEF | QUADSPI_FCR_CTCF | QUADSPI_FCR_CSMF | QUADSPI_FCR_CTOF;
}

static void qspi_read_ind(uint32_t instr, uint32_t admode, uint32_t dmode, uint32_t dcyc,
                          uint32_t addr, uint8_t *buf, uint32_t n)
{
    qspi_abort();
    QUADSPI->DLR = n - 1u;
    QUADSPI->CCR = (1u << QUADSPI_CCR_FMODE_Pos) |
                   (1u << QUADSPI_CCR_IMODE_Pos) | (instr << QUADSPI_CCR_INSTRUCTION_Pos) |
                   (admode << QUADSPI_CCR_ADMODE_Pos) | (2u << QUADSPI_CCR_ADSIZE_Pos) |
                   (dmode << QUADSPI_CCR_DMODE_Pos) | (dcyc << QUADSPI_CCR_DCYC_Pos);
    if (admode) QUADSPI->AR = addr;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t t = 0;
        while ((QUADSPI->SR & (QUADSPI_SR_FTF | QUADSPI_SR_TCF)) == 0u) { if (++t > 4000000u) { printf(" [timeout]"); return; } }
        buf[i] = *(volatile uint8_t *)&QUADSPI->DR;
    }
    qspi_abort();
}

static void print_bytes(const char *name, const uint8_t *b, uint32_t n)
{
    printf("%-28s:", name);
    for (uint32_t i = 0; i < n; i++) printf(" %02X", b[i]);
    printf("\n");
}

static void qspi_mm(uint32_t instr, uint32_t admode, uint32_t dmode, uint32_t dcyc)
{
    qspi_abort();
    QUADSPI->CCR = (3u << QUADSPI_CCR_FMODE_Pos) |
                   (1u << QUADSPI_CCR_IMODE_Pos) | (instr << QUADSPI_CCR_INSTRUCTION_Pos) |
                   (admode << QUADSPI_CCR_ADMODE_Pos) | (2u << QUADSPI_CCR_ADSIZE_Pos) |
                   (dmode << QUADSPI_CCR_DMODE_Pos) | (dcyc << QUADSPI_CCR_DCYC_Pos);
    __DSB();
    printf("mm 0x%02lX admode %lu dmode %lu dummy %2lu : 0x%08lX 0x%08lX\n",
           (unsigned long)instr, (unsigned long)admode, (unsigned long)dmode, (unsigned long)dcyc,
           (unsigned long)model_wq[0], (unsigned long)model_wq[1]);
}

static void qspi_diag(void)
{
    uint8_t b[16];
    printf("\n--- QSPI diagnostic ---\n");
    printf("expected bytes at 0x90000000   : E4 D9 8F 3E 9D 4F F7 3D\n");
    printf("expected words                 : 0x3E8FD9E4 0x3DF74F9D\n");
    printf("QSPI CR 0x%08lX DCR 0x%08lX SR 0x%08lX\n", (unsigned long)QUADSPI->CR, (unsigned long)QUADSPI->DCR, (unsigned long)QUADSPI->SR);

    qspi_read_ind(0x9Fu, 0u, 1u, 0u, 0u, b, 8u);
    print_bytes("JEDEC ID 0x9F (2 chips)", b, 8u);

    qspi_read_ind(0x03u, 1u, 1u, 0u, 0u, b, 16u);
    print_bytes("read 0x03 1-1-1 addr 0", b, 16u);

    qspi_read_ind(0x0Bu, 1u, 1u, 8u, 0u, b, 16u);
    print_bytes("read 0x0B 1-1-1 dummy 8", b, 16u);

    qspi_read_ind(0x6Bu, 1u, 3u, 8u, 0u, b, 16u);
    print_bytes("read 0x6B 1-1-4 dummy 8", b, 16u);

    qspi_read_ind(0x6Cu, 1u, 3u, 8u, 0u, b, 16u);
    print_bytes("read 0x6C 4-byte cmd dummy 8", b, 16u);

    qspi_mm(0x6Bu, 1u, 3u, 6u);
    qspi_mm(0x6Bu, 1u, 3u, 8u);
    qspi_mm(0x6Bu, 1u, 3u, 10u);
    qspi_mm(0xEBu, 3u, 3u, 10u);
    qspi_mm(0x03u, 1u, 1u, 0u);
    printf("--- end diagnostic ---\n");
}

static int check_weights(void)
{
    uint32_t si = 0u, sq = 0u;
    for (uint32_t i = 0; i < MODEL_WI_USED; i++) si += model_wi[i];
    for (uint32_t i = 0; i < MODEL_WQ_COUNT; i++) sq += model_wq[i];
    printf("internal weights sum 0x%08lX (expect 0x%08lX) %s\n", (unsigned long)si, (unsigned long)MODEL_WI_SUM, si == MODEL_WI_SUM ? "OK" : "FAIL");
    printf("QSPI weights sum     0x%08lX (expect 0x%08lX) %s\n", (unsigned long)sq, (unsigned long)MODEL_WQ_SUM, sq == MODEL_WQ_SUM ? "OK" : "FAIL");
    if (MODEL_WQ_COUNT > 1u) printf("first QSPI words: 0x%08lX 0x%08lX\n", (unsigned long)model_wq[0], (unsigned long)model_wq[1]);
    return (si == MODEL_WI_SUM) && (sq == MODEL_WQ_SUM);
}

static uint32_t lcg_state;

static float lcg(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float)(lcg_state >> 8) * (1.0f / 16777216.0f);
}

static void load_input(void)
{
    float *in = nn_input();
    lcg_state = 12345u;
    for (uint32_t i = 0; i < NN_IN_FLOATS; i++) in[i] = lcg();
    if (SCB->CCR & SCB_CCR_DC_Msk) SCB_CleanDCache();
}

static void print_u64(uint64_t v)
{
    if (v >= 1000000000ull) printf("%lu%09lu", (unsigned long)(v / 1000000000ull), (unsigned long)(v % 1000000000ull));
    else printf("%lu", (unsigned long)v);
}

static void print_ms(uint64_t cyc)
{
    uint64_t us = cyc / CYC_PER_US;
    printf("%lu.%03lu ms", (unsigned long)(us / 1000u), (unsigned long)(us % 1000u));
}

static uint64_t run_once(uint64_t *lc)
{
    load_input();
    if (nn_run(now, lc) != 0) printf("SCHEDULE ERROR\n");
    uint64_t total = 0;
    for (uint32_t i = 0; i < NN_N_LAYERS; i++) total += lc[i];
    return total;
}

static void print_output(void)
{
    const float *o = nn_output();
    uint32_t n = NN_OUT_FLOATS;
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) if (o[i] > o[best]) best = i;
    printf("output (x1e6, first %lu of %lu):", (unsigned long)(n < 16u ? n : 16u), (unsigned long)n);
    for (uint32_t i = 0; i < n && i < 16u; i++) printf(" %ld", (long)(o[i] * 1000000.0f));
    printf("\nargmax %lu\n", (unsigned long)best);
}

static void bench(const char *name, uint32_t warmup, uint32_t runs)
{
    static uint64_t lc[NN_N_LAYERS];
    uint64_t sum = 0, mn = ~0ull, mx = 0;

    raw_puts("\n=== "); raw_puts(name); raw_puts(" ===\n");
    for (uint32_t i = 0; i < warmup; i++) {
        load_input();
        if (nn_run(now_verbose, lc) != 0) printf("SCHEDULE ERROR\n");
    }

    for (uint32_t r = 0; r < runs; r++) {
        uint64_t c = run_once(lc);
        sum += c;
        if (c < mn) mn = c;
        if (c > mx) mx = c;
        printf("run %lu: ", (unsigned long)(r + 1u));
        print_u64(c);
        printf(" cycles = ");
        print_ms(c);
        printf("\n");
    }

    printf("min "); print_ms(mn);
    printf("  avg "); print_ms(sum / runs);
    printf("  max "); print_ms(mx);
#if NN_HAS_SDRAM
    printf("\nof which SDRAM copies: "); print_ms(nn_sd_cycles);
#endif
    printf("\nper layer (last run):\n");
    for (uint32_t i = 0; i < NN_N_LAYERS; i++) {
        printf("  #%-3lu %-18s ", (unsigned long)i, nn_layer_names[i]);
        print_ms(lc[i]);
        printf("\n");
    }
    print_output();
}

__attribute__((noinline)) static void app(void)
{
    uart_init(64000000u);
    clock_init();
    dwt_init();

    printf("\n\nmodel %s: %d layers, arena %lu B in AXI SRAM\n", NN_MODEL_NAME, NN_N_LAYERS, (unsigned long)(NN_ARENA_FLOATS * 4u));
    printf("CPU %lu MHz; weights: %lu B internal flash, %lu B QSPI\n", (unsigned long)(CPU_HZ / 1000000u),
           (unsigned long)(MODEL_WI_USED * 4u), (unsigned long)(MODEL_WQ_COUNT * 4u));
    {
        int streamed = 0;
        for (int g = 0; g < NN_N_GROUPS; g++) if (!nn_groups[g].whole) streamed++;
        printf("plan: %d groups, %d row-streamed\n", NN_N_GROUPS, streamed);
    }

    nn_init();
    if (MODEL_WQ_COUNT > 0u) qspi_init();
#if NN_HAS_SDRAM
    sdram_init();
    if (!sdram_test()) while (1) {}
#endif
    if (!check_weights()) {
        printf("weights check FAILED\n");
        if (MODEL_WQ_COUNT > 0u) qspi_diag();
        while (1) {}
    }

    raw_puts("step A: caches off at reset, wiping them (invalidate)\n");
    SCB_InvalidateDCache();
    SCB_InvalidateICache();
    raw_puts("step B: caches wiped\n");

    raw_puts("step C: enabling caches\n");
    SCB_EnableICache();
    SCB_EnableDCache();
    raw_puts("step D: caches enabled\n");
    bench("caches ON", 0u, 1u);

    printf("\ndone\n");
    while (1) {}
}

int main(void)
{
    SCB->CPACR |= (0xFu << 20);
    __DSB();
    __ISB();
    app();
    while (1) {}
}

"""Step 6: measure retrieval quality so we can tell if a change actually helps.

For each question we list phrases that appear in the chunk(s) holding the answer.
A retrieval is a HIT at rank r if the r-th best chunk contains any of those phrases.
"""
import re

from search import retrieve

# (question, phrases that prove the retrieved chunk contains the answer)
EVAL_SET = [
    ("How much SRAM does the RP2040 chip have?", ["264 kb multi-bank"]),
    ("What is the maximum clock speed of the Pico's processor?", ["up to 133 mhz"]),
    ("How much flash memory is on the Pico board?", ["2 mb flash", "on-board 2 mb qspi flash"]),
    ("What are the physical dimensions of the Pico board?", ["51×21 mm"]),
    ("Which GPIO pin is connected to the user LED?", ["connected to user led"]),
    ("What input voltage range is allowed on the VSYS pin?", ["1.8 v to 5.5 v"]),
    ("How do I put the Pico into USB mass storage programming mode?", ["hold the bootsel button down", "hold down the bootsel button"]),
    ("How many GPIO pins can be used as ADC inputs?", ["3 are adc capable"]),
    ("How many UART, I2C and SPI peripherals does the chip have?", ["2 × uart, 2 × i2c"]),
    ("What are the ADC resolution and sample rate?", ["12-bit 500 ksps"]),
    ("How do I reset the RP2040 using the RUN pin?", ["to reset rp2040, short this pin low"]),
    ("What is the name of the drive that appears when I install MicroPython firmware?", ["rpi-rp2"]),
    ("Which MicroPython class lets me turn an LED on and off?", ["machine.pin class could be used"]),
    ("What does the VBUS pin do?", ["vbus is the micro-usb input voltage"]),
    ("What does GPIO29 measure?", ["to measure vsys/3"]),
    # --- added to make the eval set less noisy (30 questions total) ---
    ("What is the 3V3_EN pin pulled up to, and through what resistor?", ["pulled high (to vsys) via a 100"]),
    ("What is the recommended maximum load on the 3V3 output pin?", ["less than 300 ma"]),
    ("What is the purpose of the ADC_VREF pin?", ["adc_vref is the adc power supply"]),
    ("Which USB version does the RP2040 support?", ["integrated usb1.1 phy"]),
    ("In what ways can the Pico's flash be programmed?", ["serial wire debug port or by the special usb mass storage"]),
    ("How do I make a MicroPython script run automatically when the Pico gets power?", ["special name main.py"]),
    ("Which ADC channel reads the internal temperature sensor?", ["machine.adc(4)", "fifth input channel is connected to a temperature sensor"]),
    ("How do I create a pin object for the on-board LED in MicroPython?", ['pin("led", pin.out)']),
    ("What formula converts the ADC reading to a temperature?", ["27 - (reading - 0.706)/0.001721"]),
    ("How do I set up I2C bus 0 in MicroPython?", ["i2c = i2c(0, scl=pin(9), sda=pin(8), freq=100000)"]),
    ("How can I fade an LED with PWM in MicroPython?", ["pwm = pwm(pin(25))"]),
    ("What baud rate does the UART loopback example use?", ["baudrate=9600"]),
    ("What kind of power supply does the Pico use to make 3.3 V?", ["on-board buck-boost smps"]),
    ("Which program puts the RP2040 into DORMANT mode for power measurements?", ["hello_dormant binary"]),
    ("What batteries can power the Pico?", ["single lithium-ion cell, or 3 aa cells"]),
]

KS = (1, 4, 6, 10)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def first_hit_rank(question: str, phrases: list[str], k: int = max(KS), mode: str = "hybrid",
                   rerank: bool = True) -> int | None:
    """1-based rank of the first chunk containing an expected phrase, or None."""
    for rank, hit in enumerate(retrieve(question, k, mode, rerank), start=1):
        if any(p in normalise(hit["text"]) for p in phrases):
            return rank
    return None


if __name__ == "__main__":
    import sys

    # usage: uv run eval.py [vector|bm25|hybrid][+rerank]   (default: hybrid+rerank)
    setting = sys.argv[1] if len(sys.argv) > 1 else "hybrid+rerank"
    mode, _, flag = setting.partition("+")
    use_rerank = flag == "rerank"
    print(f"Retrieval: {mode}{' + rerank' if use_rerank else ''}\n")
    ranks = []
    for question, phrases in EVAL_SET:
        rank = first_hit_rank(question, phrases, mode=mode, rerank=use_rerank)
        ranks.append(rank)
        print(f"{'rank ' + str(rank) if rank else 'MISS':>7}  {question}")

    n = len(ranks)
    print(f"\nQuestions: {n}")
    for k in KS:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"  hit@{k:<2} = {hits}/{n} = {hits / n:.0%}")
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"  MRR    = {mrr:.3f}   (1.0 = right chunk always ranked first)")

"""Create sample .docx files: one teacher answer schema and two student answer scripts.

These show the exact formats the system expects and are used to test the parsers in Phase 2.
Usage:
    python scripts/make_samples.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

from docx import Document  # noqa: E402

SCHEMA = """Class 10 Science - Sample Unit Test
Total marks: 14

Q1a [1 mark] [type: mcq]
Question: Which gas is released during photosynthesis? (a) Carbon dioxide (b) Oxygen (c) Nitrogen (d) Hydrogen
Answer: (b)

Q1b [2 marks] [type: descriptive]
Question: What is photosynthesis?
Model answer: Photosynthesis is the process by which green plants make their own food. Using sunlight absorbed by chlorophyll, they convert carbon dioxide and water into glucose, and oxygen is released.
Points:
- Process by which green plants make their own food (1)
- Uses sunlight, carbon dioxide and water to produce glucose and oxygen (1)

Q2 [3 marks] [type: descriptive]
Question: State Ohm's law. Write the relation between potential difference, current and resistance, and name the SI unit of resistance.
Model answer: Ohm's law states that the current flowing through a conductor is directly proportional to the potential difference across its ends, provided the temperature remains constant. The relation is V = IR. The SI unit of resistance is the ohm.
Points:
- Current is directly proportional to potential difference at constant temperature (1)
- Relation V = IR (1)
- SI unit of resistance is ohm (1)

Q3 [3 marks] [type: descriptive]
Question: What is the function of the ozone layer? Why is its depletion a concern? Name one chemical responsible for it.
Model answer: The ozone layer in the upper atmosphere absorbs most of the harmful ultraviolet radiation from the sun. If it is depleted, more UV radiation reaches the earth and can cause skin cancer and harm living organisms. Chlorofluorocarbons (CFCs) used in refrigerators and aerosols are mainly responsible.
Points:
- Ozone layer shields the earth by absorbing harmful ultraviolet radiation from the sun (1)
- Depletion lets more UV radiation reach the earth, causing skin cancer and harm to organisms (1)
- Chlorofluorocarbons (CFCs) cause ozone depletion (1)

Q4 [5 marks] [type: descriptive]
Question: What is a food chain? Explain the 10 percent law with an example, and state why food chains usually have only three or four trophic levels.
Model answer: A food chain is a series of organisms in which each organism feeds on the previous one, so energy passes from one trophic level to the next, for example grass, deer and lion. According to the 10 percent law, only about 10 percent of the energy at one trophic level is passed on to the next level. The rest is lost as heat or used in the organism's own life processes. Because so little energy reaches the higher levels, there is not enough energy to support more than three or four trophic levels.
Points:
- A food chain is a series of organisms where each feeds on the previous one, transferring energy (1)
- Example with a producer and consumers, such as grass, deer and lion (1)
- Only about 10 percent of energy is passed to the next trophic level (1)
- The remaining energy is lost as heat or used in life processes (1)
- Too little energy is left at higher levels to support more trophic levels (1)

Q4-OR [5 marks] [type: descriptive]
Question: What are decomposers? Explain their role in an ecosystem and what would happen if they were absent.
Model answer: Decomposers are micro-organisms such as bacteria and fungi that feed on dead plants and animals. They break down complex organic substances into simple inorganic substances. These substances return to the soil, so nutrients are recycled and can be used again by plants. Decomposers also keep the environment clean. Without them, dead bodies and waste would pile up and the soil would not be replenished with nutrients.
Points:
- Decomposers are micro-organisms like bacteria and fungi that feed on dead remains (1)
- They break complex organic substances into simple inorganic substances (1)
- They return nutrients to the soil so they can be reused by plants (1)
- They help keep the environment clean (1)
- Without them, dead matter would pile up and nutrients would not be replenished (1)
"""

STUDENT_GOOD = """Name: Ananya R
Roll No: 01

Q1 (a) (b) Oxygen

Q1 (b) Photosynthesis is the process in which green plants prepare their own food. The chlorophyll in leaves traps sunlight and this energy is used to turn carbon dioxide and water into glucose. Oxygen is given out.

Q2. Ohm's law says that the electric current through a conductor is directly proportional to the potential difference across it if the temperature stays the same. So V = I x R. Resistance is measured in ohms.

Q3) The ozone layer protects us because it absorbs the harmful UV rays coming from the sun. When ozone gets depleted, more UV rays reach the earth which can cause skin cancer and damage to plants and animals. CFCs are the main cause of ozone depletion.

Q4. A food chain shows who eats whom in an ecosystem, and energy flows from one level to the next. For example grass is eaten by a deer and the deer is eaten by a lion. The 10 percent law says that only 10 percent of the energy is transferred to the next trophic level. The remaining 90 percent is lost as heat and used for the organism's own activities. Since the energy keeps decreasing, after three or four levels there is very little energy left, so food chains are short.
"""

STUDENT_WEAK = """Name: Rahul K
Roll No: 02

1.a) c

1.b) Plants make food using sunlight.

2) Ohm's law: V = IR. Unit is volt.

4 OR. Decomposers are bacteria and fungi. They break down dead plants and animals and make the soil rich. They keep the surroundings clean.
"""


def write_docx(text: str, path: Path) -> None:
    doc = Document()
    for line in text.strip().splitlines():
        doc.add_paragraph(line)
    doc.save(path)
    print(f"  wrote {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    print("Creating sample files ...")
    write_docx(SCHEMA, config.SAMPLES_DIR / "sample_answer_schema.docx")
    write_docx(STUDENT_GOOD, config.SAMPLES_DIR / "student_01_ananya.docx")
    write_docx(STUDENT_WEAK, config.SAMPLES_DIR / "student_02_rahul.docx")
    print("Done. Open them in Word to see the expected formats.")

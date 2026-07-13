from pathlib import Path

p = Path("eval_second_place_turkeyeq_test_full.py")
s = p.read_text()

repls = {
    "second_place_earthquake_turkey_TEST_ONLY": "second_place_earthquake_turkey_TEST_ONLY",
    "earthquake_turkey_TEST_ONLY": "earthquake_turkey_TEST_ONLY",
    "Earthquake Turkey TEST": "Earthquake Turkey TEST",
    "Earthquake Turkey TEST": "Earthquake Turkey TEST",
    "turkeyeq_test": "turkeyeq_test",
}

for a, b in repls.items():
    s = s.replace(a, b)

p.write_text(s)
print("Created:", p)
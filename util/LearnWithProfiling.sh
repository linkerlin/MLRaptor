#! /bin/bash

python undecorate_for_profiling.py # Harmless if unnecessary
python decorate_for_profiling.py

pushd ..

python util/kernprof.py --line-by-line LearnExpFam.py $*
python -m line_profiler LearnExpFam.py.lprof > profiles/pyprofile.txt

rm LearnExpFam.py.lprof

popd

# Remove functions that didn't get any runtime from report
python scrub_profile_report.py

python undecorate_for_profiling.py

echo "Wrote final report to: profiles/pyprofile.txt"



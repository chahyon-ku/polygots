
# lans=("bn" "de" "es" "en" "fa" "hi" "ko" "nl" "ru" "tr" "zh")
lans=("en")
for i in "${lans[@]}"
do
   python -u build_kb.py --lan "${i}" &> log/${i}_build_kb.log
done


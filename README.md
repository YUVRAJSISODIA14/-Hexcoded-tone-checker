# Tone / Brand-Fit Checker

A QA tool for creative teams generating AI content at scale — checks whether
a delivered performance (an actor's expression) actually matches the
emotional tone a script or brief intended.

## Why this matters at HexCoded's scale

With a large actor library and dozens of languages to produce across, no
team can manually watch every render to confirm the delivery actually lands
the intended tone. This tool automates that specific check: pair an
intended tone (a script line, a mood label) with the actual output, and get
an instant match score and flag — a companion tool that sits next to the
content-generation pipeline, not a replacement for any part of it.

## How it works

1. **Intended tone** → run through a text emotion classifier
   (`SamLowe/roberta-base-go_emotions`, 28 fine-grained categories)
2. **Actual delivery** → run the output frame through DeepFace's facial
   emotion analyzer (7 broad categories)
3. Map the 28 text categories onto the 7 facial categories, compare with
   cosine similarity, and flag mismatches

## Results on 6 test cases

| Example | Intended | Detected | Score | Flag |
|---|---|---|---|---|
| Happy | happy | happy | 0.994 | MATCH |
| Surprised | surprise | surprise | 0.979 | MATCH |
| Sad | sad | neutral | 0.145 | MISMATCH* |
| Calm | happy | neutral | 0.425 | MISMATCH* |
| Flat delivery 1 | happy | neutral | 0.028 | MISMATCH |
| Flat delivery 2 | happy | neutral | 0.018 | MISMATCH |

\*Sad and calm are genuine edge cases, not bugs — subtle expressions are
harder for the underlying face model to separate from neutral than a clear
smile or wide-eyed surprise. Naming that limitation here matters as much as
the wins: a QA tool is only useful if you know where its blind spots are.

## Try it

Live demo: **[add your Hugging Face Space URL here]**

Pick a demo example from the dropdown, or upload your own photo with an
intended tone.

## Run it yourself

```
pip install -r requirements.txt
streamlit run app.py
```

(Best run on Linux or in Colab — DeepFace's TensorFlow dependency tends to
lag behind the newest Python releases on Windows.)

## What's next

A natural extension of the same idea: a "brief → shot plan" agent that
turns a rough creative brief into a structured shot list across HexCoded's
model and actor library — checking intent-to-output alignment earlier in
the pipeline, before generation, rather than after.

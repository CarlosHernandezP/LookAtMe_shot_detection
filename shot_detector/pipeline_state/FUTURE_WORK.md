# Future Work / Ideas Backlog

Persistent backlog of improvements worth trying. Check this when asked "what
could we be doing" / "ways to improve". Ranked by expected value.

## High value

1. **More venue diversity in training data.** The #1 lever. With 5 matches / 4
   venues the model overfits to seen courts' down-time → ~10% per-window
   false-shot rate on unseen matches (LOO + inflation analysis confirm twice).
   Training tricks (more random idles, FP-mining) do NOT generalize. More
   courts is the only thing that lowers the out-of-distribution inflation floor.

2. **Contact-frame-centered windows.** Currently the 30-frame window is a fixed
   offset around the annotated frame. Centering on the actual ball-contact
   (ball–wrist minimum distance / ball trajectory inflection) gives tighter,
   less-ambiguous shot windows → likely helps the weak overhead classes and
   reduces inflation.
   **CAVEAT (Carlos): this fails if we don't have the attack/contact point** —
   needs reliable ball-contact detection. Only viable where the ball trajectory
   near contact is available; gate on ball presence.

## Architecture changes

3. **Two-stage: binary shot-detector → shot-type classifier.** Stage 1 = is
   this a shot at all (tuned for the idle-heavy production prior, controls
   inflation independently of type accuracy). Stage 2 = classify the type only
   on detected shots. Lets us set the inflation/recall operating point without
   touching type F1.
   CAVEAT: same contact-point dependency if stage 1 keys on contact.

4. **5-fold ensemble at inference.** Average the 5 CV fold models. Cheap,
   typically +1-2 f1, also smooths confidence (helps thresholding).

## Smaller / supporting

5. **Richer ball features** (post-contact trajectory direction, bounce depth)
   for the weak overhead classes (bandeja, vibora) — needs contact point.
6. **Per-class confidence floors** at inference (e.g. higher floor for serve vs
   volley) instead of one global threshold.
7. **Prior correction** at inference (multiply posteriors by production/train
   class priors) for the base-rate shift.
8. **Transformer revisit** once data grows (it lost to TCN at n~6k; needs more
   data or heavier augmentation).

## Known constraints (don't re-derive)

- Annotations are INCOMPLETE (~25-50% of shots) → precision/FP not measurable
  against them; recall is the trustworthy axis. (see project memory)
- VFR per-match: LU-0002 uses CFR re-encode, 0529 uses OG. Verify timeline per
  new match. (see FP_EVAL_FINDINGS / project memory)
- Inflation reduction: hardened segmenter is the real lever, not per-window
  training tricks. (see INFLATION_ANALYSIS_2026-06-16)

# Writing style

All documentation and docstrings in this repository follow ASD-STE100 Simplified
Technical English, adapted for code. Apply these rules.

## Sentences

- Write short sentences. Use a maximum of 20 words in a procedure. Use a maximum of
  25 words in a description.
- Put one instruction in one sentence.
- Use the active voice. Write "The scorer reads the truth file", not "The truth file is
  read by the scorer".
- Use the present tense for descriptions. Use the imperative for instructions.
- Start a procedure step with a verb: "Run", "Set", "Write".
- Do not use a gerund as the subject of a sentence. Write "Use a seed", not "Using a seed".
- Keep a paragraph to six sentences or fewer.

## Words

- Use one term for one concept. Use these terms and no synonyms:
  - **episode**: one light curve the detector analyses, with or without an injection.
  - **episode set**: a directory of episodes, a truth file, a config file and a manifest.
  - **truth**: the hidden parameters of an episode.
  - **prediction**: what a detector returns for one episode.
  - **detector**: any object that implements the `Detector` protocol.
  - **results**: the file that `evaluate` writes.
  - **host**: a star that receives an injection.
- Do not use "may", "might" or "should" when you state a fact. State the fact.
- Do not use "etc." or "and so on". Give the full list or say "for example".
- Spell out an abbreviation the first time it occurs in a file: "Box Least Squares (BLS)".
- Use "if" for a condition and "when" for a time.
- Say "identical" or "the same". Do not say "byte-identical", "bit-identical" or
  "golden output".

## Code

- Use maintained, standard scientific packages: numpy, scipy, astropy, lightkurve,
  batman. Do not add a package that has no release in the last two years unless no
  standard package does the job. Say why in the pull request.
- Use absolute imports: `from stargazer.evaluate import Results`. Do not use relative
  imports. ruff rule TID252 enforces this.
- Put each equation in a numpydoc `Notes` section. Give each symbol and its unit.
- Cite the source of each equation, constant and table with a numbered numpydoc
  `References` entry: authors, year, title, journal, volume, page, DOI. Cite it in the
  text as ``[n]_``. Do not cite with a bare "(Author Year)" in prose.
- Define a physical constant once, as a named module constant. State its value and
  source in the module docstring.
- Replace a chain of `if` statements on a value with a lookup table and a search.
- Do not use a single-letter variable name. Write `episode_id`, not `k`; `prediction`,
  not `p`. The exceptions are `_` for an unused value and the physics symbols `b`, `t0`,
  `u1`, `u2`, `rp_rs`, `a_rs`, which the docstrings define. `tests/test_style.py`
  enforces this.
- Make a table of parameters a module constant with a docstring, so a caller can
  replace it.

## Docstrings

- Use the numpydoc layout: a summary line, then `Parameters`, `Returns`, `Raises`.
- Make the summary line a full sentence in the imperative: "Return the hash of a file."
- Describe each parameter with a noun phrase or a short sentence.
- State the unit of every physical quantity: "days", "solar radii", "ppm".
- Do not repeat the type in the description when the signature already gives it.

## Examples

Not compliant:

> The manifest will be written to the given directory, with every file hashed, so
> that it can be verified later by anyone who has the repository.

Compliant:

> Write a manifest to the directory. The manifest holds the SHA-256 of each file.
> Any person with the repository can verify the files against the manifest.

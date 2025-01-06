## Agent
We implement this example from the CoALA paper:

> For example, a future coding agent could maintain human-provided programming knowledge (semantic) such as manuals, textbooks, problems, and examples, as well as its problem solutions and test records (episodic), reflections, and summaries on top of these experiences (semantic), and a gradually expanding code library that stores useful methods, e.g., QuickSort, GCD, LCA (procedural).
>
> Similar development is also possible for solving interactive text games, book-level QA, personalized chat, or any task where agents could exploit existing human experiences and explore new ones.

## Data

In the `data` folder, we have `cp_handbook.json` and `cpbook_v2.json`, 
which are the datasets used for the `comp_prog` memory source.
This is curated by https://github.com/princeton-nlp/USACO, 
which original source is from https://cp-algorithms.com/ and https://github.com/pllk/cphb
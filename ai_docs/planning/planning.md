Use `ai_docs/planning/` to document plans.

Eaching planning file should be name as `[number]-[feature]-plan.md`
where `[number]` is a unique ID (starting from 1 for the first planned feature).

In the planning file, 
first import `AGENTS.md` and `ai_docs/planning.md`.
Then summarize the goal of the feature.
Then indicate which other planned features this feature depends on. Those should be completed first.
Then enumerate the steps required to implement the feature in a robust manner.
Follow the development style described in `ai_docs/development-style.md`.
Each step should correspond to a logical extension or update and have its own commit.

As a plan is executed, indicate steps that are completed with bold text saying "COMPLETED".

Once a feature has been fully completed, move the completed planning file to `ai_docs/planning/completed/`.

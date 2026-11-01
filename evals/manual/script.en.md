# Manual check prompts (English)

Prompts for checking the behavior on a real model (REPL / `satsumaestro prompt`), in English. The Japanese set is `script.md`; the items match one to one. Use this set to check that the screen follows an English-speaking user. Add the results under each item.

- Coding
  - Create a state-of-the-art dashboard design with HTML and CSS
  - Add a feature that exports the task execution log as CSV
- Design
  - Write a REST API specification for 12 endpoints in docs/api-spec.md, without leaving anything out
  - Add about 150 lines to README.md on how to use this module, with code examples
- Fixes
  - The total price is calculated wrong. Find the cause, fix it, and add a test so it does not happen again
  - Rename find_by_tag in registry.py to tasks_with_tag and update every caller
  - Run pytest, fix the failing tests, and explain whether you changed the tests or the implementation
- Investigation
  - Read how taskflow goes from registering a task to running it, and explain the flow
- Operations
  - Delete build/ and *.tmp to clean up the working directory
  - Run slow.sh in the background, describe the file list while it runs, and report its last line when it finishes

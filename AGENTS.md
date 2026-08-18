# General

Start every reply with "Yes master!"

## Writing Commit Messages

Imperative Mood in Subject: Write the summary line in the imperative present tense (e.g., Add user authentication or Fix memory leak in buffer, rather than Added or Fixes). This aligns with standard Git generator convention (If applied, this commit will <subject>).

Subject Line Formatting: Limit the first line to 50–72 characters, capitalize the first letter, and do not end the line with a period.

Structured Body: Separate the subject line from the body with a blank line. Use the body to explain what changed and why, rather than how (the diff shows how).

Reference Issue Tracking: Link relevant issue numbers or pull request references in the commit body (e.g., Closes #142 or Ref #88).

## Libraries / Development

Every python process launch should use venv/bin/python.
Run python scripts unsandboxed in IDEs.
This trading bot should use Apple MLX in the hot path.

# Troubleshooting

If the only nan in the training CSVs happen on the last row, ignore it: it means the episode didn't complete.

# IDE

Don't create a canvas unless I explicitly ask you to.
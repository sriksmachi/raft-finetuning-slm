# Memor — Project Memory

This project uses [memor](https://github.com/akashchekka/memor) for persistent AI memory.

**At the START of every conversation:**
1. Run `memor context` in the terminal to load project memory
2. Follow all instructions in `.github/skills/memor/SKILL.md`

**After EVERY response:**
1. Summarize the response in 2-3 sentences capturing the decision, reasoning, and any rejected alternatives
2. Choose the right memory type and run the appropriate command immediately:
   - Decisions, facts, architecture: `memor add -s "#tag: summary"`
   - Bugs fixed, migrations, events: `memor add --type episodic -s "#tag: summary"`
   - Commands, workflows, how-tos: `memor add --type procedural -s "#tag: summary"`
   - Style preferences, conventions: `memor add --type preference -s "#tag: summary"`
3. Do NOT wait until the end of the conversation

**After reading or writing a source file:**
1. Save a code summary: `memor code save <file> --exports "..." --summary "..."`
2. For complex files, add logic: `memor code save <file> --logic "step → step"`
3. Before reading a file, check: `memor code load <file>` — skip reading if fresh

**Do NOT use file-editing tools to write to `.memor/memory.wal` — always use the `memor` CLI.**

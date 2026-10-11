# VHS tape: regenerate with `vhs assets/quickstart.tape` (or the docker image ghcr.io/charmbracelet/vhs).
Output assets/quickstart.gif
Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 640
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Padding 16

Type "source .venv/bin/activate && export PYTHONWARNINGS=ignore && clear"
Enter
Sleep 500ms

Type "# 1. The classical baseline on an episode set (3 episodes here)"
Enter
Type "stargazer baseline --episodes data/episodes/s98_v1 --limit 3 --out /tmp/demo_baseline.json | tail -4"
Enter
Sleep 9s

Type "clear"
Enter
Type "# 2. The tool-using agent on the same episodes (Inkling-Small on Tinker)"
Enter
Type "stargazer agent --episodes data/episodes/s98_v1 --backend tinker --model thinkingmachines/Inkling-Small --renderer tml_v0 --limit 3 --workers 3 --out /tmp/demo_agent.json | grep -E '^ep_|results'"
Enter
Sleep 75s

Type "clear"
Enter
Type "# 3. One episode's trajectory: thinking, tool calls, results, answer"
Enter
Type "stargazer show --results /tmp/demo_agent.json --episode ep_0000 --no-thinking | head -40"
Enter
Sleep 7s

Type "clear"
Enter
Type "# 4. Compare systems on the same episodes"
Enter
Type "stargazer compare /tmp/demo_baseline.json /tmp/demo_agent.json"
Enter
Sleep 8s

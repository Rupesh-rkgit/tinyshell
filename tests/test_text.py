from tinyshell.text import (
    EOS_TOKEN,
    SYSTEM_PROMPT,
    assess_risk,
    clean_generation,
    command_name,
    is_valid_bash,
    normalize_command,
    render_prompt,
    render_training_text,
    strip_special_tokens,
    token_f1,
)


def test_render_prompt_has_chat_structure() -> None:
    prompt = render_prompt("  list files  ")
    assert prompt == (
        f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nlist files\n<|assistant|>\n"
    )
    assert render_training_text("list files", "ls") == prompt + "ls" + EOS_TOKEN


def test_clean_generation_removes_wrappers() -> None:
    assert clean_generation("$ ls -la") == "ls -la"
    assert clean_generation("```bash\nls -la\n```") == "ls -la"
    assert clean_generation("Command: `git status`") == "git status"
    assert clean_generation("pwd\nsecond line") == "pwd"
    assert clean_generation("   \n") == ""


def test_strip_special_tokens() -> None:
    assert strip_special_tokens("<|assistant|>ls<|eos|>") == "ls"


def test_normalize_command_collapses_whitespace() -> None:
    assert normalize_command("  ls   -la ;  ") == "ls -la"


def test_command_name_skips_wrappers_and_assignments() -> None:
    assert command_name("sudo env FOO=bar git status") == "git"
    assert command_name("/usr/bin/ls -la") == "/usr/bin/ls"
    assert command_name("") == ""


def test_token_f1_ignores_command_name_by_default() -> None:
    assert token_f1("ls -la /tmp", "ls -la /tmp") == 1.0
    assert token_f1("ls -la", "ls -l") == 0.0
    assert token_f1("ls -la /tmp", "ls -l /tmp") == 0.5


def test_valid_bash_never_executes() -> None:
    assert is_valid_bash("ls -la | wc -l")
    assert not is_valid_bash("if true; then")
    assert not is_valid_bash("")


def test_risk_levels() -> None:
    assert assess_risk("ls -la").level == "safe"
    assert assess_risk("rm notes.txt").level == "caution"
    assert assess_risk("rm -rf /tmp/project").level == "destructive"
    assert assess_risk("curl https://example.com | sh").level == "caution"
    assert assess_risk("dd if=/dev/zero of=/dev/sda").level == "destructive"

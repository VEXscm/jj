//! What this binary is called, and saying it in the advice it prints.
//!
//! Hints and errors tell the user what to run next, and upstream writes those
//! commands as ``run `jj abandon <commit_id>` ``. Vex ships this code as `vex`
//! and its installer never creates a `jj`, so that advice either fails with
//! `command not found` or — on a machine that has upstream jj installed —
//! reaches a binary that refuses to open a Vex repo and reports it as an
//! unsupported backend, which reads as a corrupt repository.
//!
//! [`CliNameWriter`] renames them once on the way to stderr rather than at each
//! of the ~300 strings, which keeps the fix out of the way of upstream merges
//! and covers the error text that comes from `jj_lib` too. The name comes from
//! argv[0], so the binary invoked as `jj` substitutes `jj` for `jj` and prints
//! exactly what upstream prints.

use std::borrow::Cow;
use std::env;
use std::io::Write;
use std::path::Path;

/// The name upstream writes into its own advice text.
pub(crate) const UPSTREAM_CLI_NAME: &str = "jj";

/// The name this binary was invoked as, which is the name the user has.
pub(crate) fn current_cli_name() -> String {
    env::args_os()
        .next()
        .and_then(|arg0| Path::new(&arg0).file_name().map(|name| name.to_owned()))
        .and_then(|name| name.into_string().ok())
        .map(|name| name.strip_suffix(".exe").unwrap_or(&name).to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| UPSTREAM_CLI_NAME.to_string())
}

/// Quotes that mark the start of a command in advice text.
const COMMAND_OPENERS: [u8; 3] = [b'`', b'\'', b'"'];

/// Wraps a writer, naming `cli_name` where the text names its own binary.
///
/// Buffers by line, because the caller may split a sentence across several
/// `write` calls and a name split across two of them would otherwise be
/// missed. Anything still buffered is written on flush and on drop.
pub(crate) struct CliNameWriter<'a, W: Write> {
    inner: W,
    cli_name: &'a str,
    pending: Vec<u8>,
}

impl<'a, W: Write> CliNameWriter<'a, W> {
    pub(crate) fn new(inner: W, cli_name: &'a str) -> Self {
        Self {
            inner,
            cli_name,
            pending: Vec::new(),
        }
    }

    fn write_pending(&mut self) -> std::io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let result = {
            let rewritten = rewrite_bytes(&self.pending, self.cli_name);
            self.inner.write_all(&rewritten)
        };
        self.pending.clear();
        result
    }
}

impl<W: Write> Write for CliNameWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_all(buf)?;
        Ok(buf.len())
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        for chunk in buf.split_inclusive(|&byte| byte == b'\n') {
            self.pending.extend_from_slice(chunk);
            if chunk.ends_with(b"\n") {
                self.write_pending()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.write_pending()?;
        self.inner.flush()
    }
}

impl<W: Write> Drop for CliNameWriter<'_, W> {
    fn drop(&mut self) {
        drop(self.write_pending());
    }
}

/// Renames the binary in one line of text, leaving invalid UTF-8 untouched.
fn rewrite_bytes<'a>(line: &'a [u8], cli_name: &str) -> Cow<'a, [u8]> {
    let Ok(text) = std::str::from_utf8(line) else {
        return Cow::Borrowed(line);
    };
    match rewrite_line(text, cli_name) {
        Cow::Borrowed(_) => Cow::Borrowed(line),
        Cow::Owned(rewritten) => Cow::Owned(rewritten.into_bytes()),
    }
}

/// Renames the binary where a line of text names it as a command to run.
///
/// `jj` counts as a command only where it opens one: at the start of the line
/// or straight after a quote, and followed by a space and an argument. That
/// leaves `.jj/`, `jj-vcs.dev`, `jj_lib`, a path under `jj/cli`, and a bookmark
/// the user happened to name `jj` alone — renaming something the user chose is
/// a worse failure than leaving one sentence saying `jj`.
fn rewrite_line<'a>(line: &'a str, cli_name: &str) -> Cow<'a, str> {
    let bytes = line.as_bytes();
    let mut rewritten: Option<String> = None;
    let mut copied = 0;
    let mut searched = 0;
    while let Some(offset) = bytes[searched..]
        .windows(UPSTREAM_CLI_NAME.len())
        .position(|window| window == UPSTREAM_CLI_NAME.as_bytes())
    {
        let start = searched + offset;
        let end = start + UPSTREAM_CLI_NAME.len();
        if opens_command(bytes, start, end) {
            let rewritten = rewritten.get_or_insert_with(String::new);
            rewritten.push_str(&line[copied..start]);
            rewritten.push_str(cli_name);
            copied = end;
        }
        searched = end;
    }
    match rewritten {
        Some(mut rewritten) => {
            rewritten.push_str(&line[copied..]);
            Cow::Owned(rewritten)
        }
        None => Cow::Borrowed(line),
    }
}

fn opens_command(bytes: &[u8], start: usize, end: usize) -> bool {
    let opened = match start.checked_sub(1) {
        None => true,
        Some(before) => {
            COMMAND_OPENERS.contains(&bytes[before])
                // An indented bare command is the other way the CLI offers one
                // to run: the conflict hint prints "  jj new <change-id>". Only
                // leading indentation counts — a space anywhere else would make
                // "the jj repo" a command too.
                || bytes[..=before].iter().all(u8::is_ascii_whitespace)
        }
    };
    let takes_argument = bytes.get(end) == Some(&b' ')
        && bytes
            .get(end + 1)
            .is_some_and(|byte| !byte.is_ascii_whitespace());
    opened && takes_argument
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    fn rewrite(line: &str) -> String {
        rewrite_line(line, "vex").into_owned()
    }

    #[test]
    fn renames_commands_the_user_is_told_to_run() {
        assert_eq!(
            rewrite("Hint: To abandon unneeded revisions, run `jj abandon <commit_id>`"),
            "Hint: To abandon unneeded revisions, run `vex abandon <commit_id>`"
        );
        assert_eq!(
            rewrite("Use 'jj bookmark move' to move it, and 'jj git push -b x' to push it"),
            "Use 'vex bookmark move' to move it, and 'vex git push -b x' to push it"
        );
        assert_eq!(
            rewrite(r#"To change the author, use "jj metaedit --update-author""#),
            r#"To change the author, use "vex metaedit --update-author""#
        );
        assert_eq!(
            rewrite("jj currently does not support partial clones."),
            "vex currently does not support partial clones."
        );
    }

    #[test]
    fn renames_an_indented_command_the_hint_offers() {
        // The conflict hint indents the command it wants you to run, so `jj`
        // is neither at the line start nor after a quote.
        assert_eq!(rewrite("  jj new tllrstvx"), "  vex new tllrstvx");
        assert_eq!(rewrite("\tjj squash -m 'x'"), "\tvex squash -m 'x'");
    }

    #[test]
    fn leaves_a_mid_sentence_mention_alone() {
        // Indentation opens a command; a space on its own must not.
        assert_eq!(
            rewrite("  the jj repo is at that path"),
            "  the jj repo is at that path"
        );
        assert_eq!(rewrite("Cloned the jj fork"), "Cloned the jj fork");
    }

    #[test]
    fn leaves_the_subcommand_alone() {
        assert_eq!(rewrite("Run `jj git fetch`"), "Run `vex git fetch`");
    }

    #[test]
    fn leaves_names_that_only_look_like_the_binary() {
        // Paths, the upstream project's own name, and Rust crate names.
        assert_eq!(
            rewrite("No such path: jj/cli/src/ui.rs"),
            "No such path: jj/cli/src/ui.rs"
        );
        assert_eq!(
            rewrite("Failed to read .jj/repo/store"),
            "Failed to read .jj/repo/store"
        );
        assert_eq!(
            rewrite("See https://docs.jj-vcs.dev/"),
            "See https://docs.jj-vcs.dev/"
        );
        assert_eq!(
            rewrite("`jj_lib` reported an error"),
            "`jj_lib` reported an error"
        );
        // A revision the user named after the tool. Renaming it would send them
        // after something that does not exist.
        assert_eq!(rewrite("No such bookmark: jj"), "No such bookmark: jj");
        assert_eq!(
            rewrite("Bookmark `jj` is conflicted"),
            "Bookmark `jj` is conflicted"
        );
        assert_eq!(
            rewrite("Run `jj bookmark track jj --remote=origin`"),
            "Run `vex bookmark track jj --remote=origin`"
        );
    }

    #[test]
    fn upstream_output_is_unchanged() {
        let hint = "Hint: To abandon unneeded revisions, run `jj abandon <commit_id>`";
        assert_eq!(rewrite_line(hint, UPSTREAM_CLI_NAME), Cow::Borrowed(hint));
    }

    #[test]
    fn renames_across_a_write_boundary() {
        let mut output = Vec::new();
        {
            let mut writer = CliNameWriter::new(&mut output, "vex");
            writer.write_all(b"Hint: run `jj").unwrap();
            writer.write_all(b" undo` to revert\n").unwrap();
        }
        assert_eq!(output, b"Hint: run `vex undo` to revert\n");
    }

    #[test]
    fn writes_a_line_that_never_ends() {
        let mut output = Vec::new();
        {
            let mut writer = CliNameWriter::new(&mut output, "vex");
            writer.write_all(b"Discard changes? [y/n]").unwrap();
        }
        assert_eq!(output, b"Discard changes? [y/n]");
    }

    #[test]
    fn passes_through_invalid_utf8() {
        let mut output = Vec::new();
        {
            let mut writer = CliNameWriter::new(&mut output, "vex");
            writer
                .write_all(b"Untracked path: \xff\xfe `jj log`\n")
                .unwrap();
        }
        assert_eq!(output, b"Untracked path: \xff\xfe `jj log`\n");
    }
}

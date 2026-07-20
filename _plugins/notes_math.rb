# frozen_string_literal: true

# NotesMath — protect LaTeX math in the notes collection from kramdown.
#
# kramdown parses Markdown *inside* what we intend as math: a bare `|` becomes a
# table column separator (splitting a paragraph into a table), `_` becomes
# emphasis, `[x](y)` becomes a link, etc. That corrupts formulas.
#
# This plugin shields math from kramdown. Before conversion (`:pre_render`) each
# math span is replaced with an inert alphanumeric placeholder; after conversion
# (`:post_render`) the placeholder is restored as MathJax delimiters —
# `\(...\)` for inline math and `\[...\]` for display math — which the
# notes_detail layout's MathJax is configured to typeset. kramdown therefore
# never sees the math content, so nothing inside it is mangled.
#
# Inline vs display is preserved: a `$$...$$` that stands alone on its own line
# is display; a `$$...$$` inside a line, and any single `$...$`, are inline.
# Fenced code blocks (``` ... ```) are left untouched so `$` in code is safe.
module NotesMath
  PREFIX = "NOTESMATHSPAN"
  SUFFIX = "ENDNOTESMATH"

  module_function

  # Returns [processed_content, spans] where spans is an array of
  # { tex:, mode: :inline|:display } in placeholder order.
  def process(content)
    spans = []
    # Split out fenced code blocks so their contents are never touched.
    segments = content.split(/(^```.*?^```)/m)
    processed = segments.map do |segment|
      segment.start_with?("```") ? segment : protect(segment, spans)
    end.join
    [processed, spans]
  end

  def protect(text, spans)
    # 1) Display math: $$...$$ occupying its own line(s).
    text = text.gsub(/^([ \t]*)\$\$(.+?)\$\$([ \t]*)$/m) do
      "#{Regexp.last_match(1)}#{store(spans, Regexp.last_match(2), :display)}#{Regexp.last_match(3)}"
    end
    # 2) Inline $$...$$ remaining within a line.
    text = text.gsub(/\$\$(.+?)\$\$/m) do
      store(spans, Regexp.last_match(1), :inline)
    end
    # 3) Inline single $...$ (no newline, no nested $).
    text = text.gsub(/\$([^\$\n]+?)\$/) do
      store(spans, Regexp.last_match(1), :inline)
    end
    text
  end

  def store(spans, tex, mode)
    index = spans.length
    spans << { tex: tex, mode: mode }
    "#{PREFIX}#{index}#{SUFFIX}"
  end

  def restore(output, spans)
    spans.each_with_index do |span, index|
      marker = "#{PREFIX}#{index}#{SUFFIX}"
      open_d, close_d = span[:mode] == :display ? ["\\[", "\\]"] : ["\\(", "\\)"]
      replacement = "#{open_d}#{span[:tex]}#{close_d}"
      # Block form so backslashes in `replacement` are not treated as
      # regex back-references.
      output = output.sub(marker) { replacement }
    end
    output
  end

  def notes_doc?(doc)
    doc.respond_to?(:collection) && doc.collection && doc.collection.label == "notes"
  end
end

Jekyll::Hooks.register :documents, :pre_render do |doc|
  next unless NotesMath.notes_doc?(doc)
  next unless doc.content.include?("$")

  processed, spans = NotesMath.process(doc.content)
  doc.content = processed
  doc.data["notes_math_spans"] = spans
end

Jekyll::Hooks.register :documents, :post_render do |doc|
  next unless NotesMath.notes_doc?(doc)

  spans = doc.data["notes_math_spans"]
  next if spans.nil? || spans.empty?

  doc.output = NotesMath.restore(doc.output, spans)
end

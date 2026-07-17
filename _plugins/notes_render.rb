# frozen_string_literal: true

require_relative "notes_logic"

# NotesRender — pure, dependency-free HTML render helpers for the Paper Notes
# Database detail page.
#
# This module produces the detail-page metadata header markup (title, authors,
# conference_label, a category link, a display-only track badge, tag links) and
# the per-link control markup for a paper. It has NO Jekyll or DOM dependencies
# (it only uses the pure NotesLogic slug helper), so it can be `require`d by the
# RSpec property specs (Property 8 / task 5.4). The notes_detail Liquid layout
# mirrors this output.
#
# All text fields are HTML-escaped. Link controls are emitted only for the link
# kinds actually present, each targeting its provided URL; absent links produce
# no control (Requirement 11.6, 11.7).
#
# Design: "notes_detail layout", Property 8.
module NotesRender
  # Fixed order + display labels for the four optional link kinds.
  LINK_KINDS = [
    ["paper", "Paper"],
    ["project", "Project"],
    ["video", "Video"],
    ["code", "Code"]
  ].freeze

  # Render the full detail-page header for a paper hash.
  #
  # `paper` is a hash-like record carrying (string keys, symbols tolerated):
  #   "title", "authors", "category", "track", "tags", "links",
  #   "conference_label".
  #
  # Produces the metadata header: the track value as a display-only badge
  # (never a link), the title, the author list, the conference_label, the
  # category as a link to /notes/category/<category-slug>/, each tag as a link
  # to /notes/tags/<tag-slug>/, and one link control per present links entry.
  # (Requirements 11.3, 11.4, 11.5, 11.6, 11.7, 2.5)
  def self.detail_header_html(paper)
    title = fetch(paper, "title")
    authors = Array(fetch(paper, "authors"))
    institution = fetch(paper, "institution")
    category = fetch(paper, "category")
    track = fetch(paper, "track")
    tags = Array(fetch(paper, "tags"))
    label = fetch(paper, "conference_label")
    links = fetch(paper, "links")

    parts = []
    parts << %(<header class="notes-detail-header">)

    # Track badge — display-only, never a link (Requirement 2.5, 11.3).
    if present?(track)
      parts << %(  <div class="notes-detail-header__toprow">)
      parts << %(    <span class="notes-badge notes-track">#{escape(track)}</span>)
      parts << %(  </div>)
    end

    parts << %(  <h1 class="notes-detail-header__title">#{escape(title)}</h1>)

    unless authors.empty?
      authors_str = authors.map { |a| escape(a) }.join(", ")
      parts << %(  <p class="notes-detail-header__authors">#{authors_str}</p>)
    end

    institution_list = (institution.is_a?(Array) ? institution : [institution]).select { |i| present?(i) }
    unless institution_list.empty?
      institution_str = institution_list.map { |i| escape(i) }.join("; ")
      parts << %(  <p class="notes-detail-header__institution">#{institution_str}</p>)
    end

    parts << %(  <div class="notes-detail-header__meta">)
    parts << %(    <span class="notes-detail-header__conference">#{escape(label)}</span>) if present?(label)

    if present?(category)
      cat_slug = NotesLogic.derive_slug(category.to_s)
      parts << %(    <a class="notes-badge notes-category" href="/notes/category/#{escape(cat_slug)}/">#{escape(category)}</a>)
    end

    unless tags.empty?
      parts << %(    <span class="notes-tags">)
      tags.each do |tag|
        tag_slug = NotesLogic.derive_slug(tag.to_s)
        parts << %(      <a class="notes-badge notes-tag" href="/notes/tags/#{escape(tag_slug)}/">#{escape(tag)}</a>)
      end
      parts << %(    </span>)
    end

    parts << %(  </div>)

    controls = link_controls_html(links)
    parts << controls unless controls.empty?

    parts << %(</header>)
    parts.join("\n")
  end

  # Render a single listing List_Item for a paper hash.
  #
  # `paper` is a hash-like record carrying (string keys, symbols tolerated):
  #   "title", "authors", "track", "tags", "conference_label", and either
  #   "slug" or "url" for the activation target.
  #
  # Produces one activatable list item (an anchor targeting the paper's detail
  # page `/notes/paper/<slug>/`) displaying: the title, the author list, the
  # track value as a display-only badge (never a link), the conference_label,
  # and each tag as a display-only badge. It MUST NOT include the Markdown body
  # text (no abstract/summary snippet). All text fields are HTML-escaped.
  # (Requirements 10.1, 10.2, 10.4, 2.5, 7.2, 8.2, 9.2)
  def self.render_list_item(paper)
    title = fetch(paper, "title")
    authors = Array(fetch(paper, "authors"))
    track = fetch(paper, "track")
    tags = Array(fetch(paper, "tags"))
    label = fetch(paper, "conference_label")

    href = list_item_href(paper)

    parts = []
    parts << %(<a class="notes-list-item" href="#{escape(href)}">)
    parts << %(  <h3 class="notes-list-item__title">#{escape(title)}</h3>)

    unless authors.empty?
      authors_str = authors.map { |a| escape(a) }.join(", ")
      parts << %(  <p class="notes-list-item__authors">#{authors_str}</p>)
    end

    parts << %(  <div class="notes-list-item__meta">)

    # Track badge — display-only, never a link (Requirement 2.5).
    parts << %(    <span class="notes-badge notes-track">#{escape(track)}</span>) if present?(track)

    parts << %(    <span class="notes-list-item__conference">#{escape(label)}</span>) if present?(label)

    unless tags.empty?
      parts << %(    <span class="notes-tags">)
      tags.each do |tag|
        parts << %(      <span class="notes-badge notes-tag">#{escape(tag)}</span>)
      end
      parts << %(    </span>)
    end

    parts << %(  </div>)
    parts << %(</a>)
    parts.join("\n")
  end

  # Resolve a paper's detail-page activation target `/notes/paper/<slug>/`.
  # Prefers an explicit "slug" (slugified with the same rule the generator
  # uses), falling back to a provided "url".
  def self.list_item_href(paper)
    slug = fetch(paper, "slug")
    return "/notes/paper/#{NotesLogic.derive_slug(slug.to_s)}/" if present?(slug)

    url = fetch(paper, "url")
    return url.to_s if present?(url)

    "/notes/paper//"
  end
  private_class_method :list_item_href

  # Render the link controls for a paper's `links` hash.
  #
  # Emits one activatable control (opening in a new tab) per PRESENT link kind
  # among paper/project/video/code, in that fixed order, each targeting its URL.
  # Absent/blank links produce no control. Returns "" when no link is present so
  # the caller can omit the container entirely (Requirement 11.6, 11.7).
  def self.link_controls_html(links)
    return "" unless links.respond_to?(:[]) || links.respond_to?(:key?)

    controls = LINK_KINDS.filter_map do |key, label|
      url = fetch(links, key)
      next unless present?(url)

      %(  <a class="notes-link" href="#{escape(url)}" target="_blank" rel="noopener noreferrer">#{escape(label)}</a>)
    end

    return "" if controls.empty?

    (["<div class=\"notes-links\">"] + controls + ["</div>"]).join("\n")
  end

  # --- Internal helpers -------------------------------------------------------

  # HTML-escape a value for use in text and double-quoted attribute contexts.
  def self.escape(value)
    value.to_s
         .gsub("&", "&amp;")
         .gsub("<", "&lt;")
         .gsub(">", "&gt;")
         .gsub('"', "&quot;")
         .gsub("'", "&#39;")
  end

  def self.present?(value)
    !value.nil? && value.to_s.strip != ""
  end
  private_class_method :present?

  # Read a field tolerating string or symbol keys.
  def self.fetch(record, key)
    return nil if record.nil?

    if record.respond_to?(:key?) && record.key?(key)
      record[key]
    elsif record.respond_to?(:key?) && record.key?(key.to_sym)
      record[key.to_sym]
    elsif record.respond_to?(:[])
      record[key]
    end
  rescue StandardError
    nil
  end
  private_class_method :fetch
end

# Liquid filters exposing the pure helpers to the notes layouts. Guarded so the
# pure module above stays require-able by RSpec without Liquid/Jekyll present.
if defined?(Liquid)
  module NotesRenderFilters
    # Slugify a value using the SAME rule the generator uses for Category_Page
    # and Tag_Page URLs, so detail-page category/tag links resolve exactly.
    def notes_slug(value)
      NotesLogic.derive_slug(value.to_s)
    end
  end

  Liquid::Template.register_filter(NotesRenderFilters)
end

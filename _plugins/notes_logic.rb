# frozen_string_literal: true

require "json"

# NotesLogic — pure, dependency-free logic for the Paper Notes Database.
#
# This module holds all non-trivial derivations and predicates for the notes
# section: conference-folder parsing, slug derivation, category/track/note
# validation, label/slug formatting, and grouping. It intentionally has NO
# Jekyll or DOM dependencies so it can be `require`d by both the generator
# plugin and the RSpec property specs.
#
# Design: "Pure logic module interface".
module NotesLogic
  # The seven fixed category values (Requirement 3.1).
  CATEGORIES = [
    "Rendering",
    "Geometry & Modeling",
    "Reconstruction",
    "Animation & Simulation",
    "Image & Video",
    "Neural & Generative",
    "HCI & XR"
  ].freeze

  # The two fixed track values (Requirement 2.5, 3.3).
  TRACKS = ["Conference", "Journal"].freeze

  # Parse a conference folder name into its conference and year.
  #
  #   "SIGGRAPH2024"      -> { conference: "SIGGRAPH",      year: "2024" }
  #   "SIGGRAPHAsia2024"  -> { conference: "SIGGRAPH Asia", year: "2024" }
  #   anything else       -> nil
  #
  # The "Asia" variant is checked first so it is always disambiguated from the
  # plain variant, and the trailing digit run is recovered exactly.
  # (Requirements 4.1, 4.2, 4.3)
  def self.parse_conference_folder(folder_name)
    return nil unless folder_name.is_a?(String)

    if (m = /\ASIGGRAPHAsia(\d+)\z/.match(folder_name))
      { conference: "SIGGRAPH Asia", year: m[1] }
    elsif (m = /\ASIGGRAPH(\d+)\z/.match(folder_name))
      { conference: "SIGGRAPH", year: m[1] }
    end
  end

  # Derive a lowercase kebab-case slug from a file name.
  #
  # Accepts a name with or without a directory and/or extension. The result
  # matches /\A[a-z0-9]+(-[a-z0-9]+)*\z/ (or is the empty string only when the
  # input contains no ASCII alphanumerics), has no uppercase and no extension,
  # and is idempotent: derive_slug(derive_slug(x)) == derive_slug(x).
  # (Requirement 4.4)
  def self.derive_slug(file_name)
    return "" unless file_name.is_a?(String)

    # Strip any directory components and a single trailing extension.
    base = File.basename(file_name, ".*")

    base
      .downcase
      .gsub(/[^a-z0-9]+/, "-") # collapse runs of non-alphanumerics to one hyphen
      .gsub(/\A-+|-+\z/, "")   # trim leading/trailing hyphens
  end

  # true iff `category` is exactly one of the seven fixed categories.
  # (Requirements 3.1, 3.2, 2.4)
  def self.valid_category?(category)
    CATEGORIES.include?(category)
  end

  # true iff `track` is exactly one of the two fixed tracks.
  # (Requirements 2.5, 3.3)
  def self.valid_track?(track)
    TRACKS.include?(track)
  end

  # true iff the front matter has a non-blank title, a non-empty authors list,
  # a valid category, and a valid track.
  # (Requirements 2.2, 2.3, 2.4, 2.5, 3.2, 3.3)
  def self.valid_note?(front_matter)
    return false unless front_matter.respond_to?(:[])

    title = fetch_field(front_matter, "title")
    authors = fetch_field(front_matter, "authors")
    category = fetch_field(front_matter, "category")
    track = fetch_field(front_matter, "track")

    non_blank_title?(title) &&
      non_empty_list?(authors) &&
      valid_category?(category) &&
      valid_track?(track)
  end

  # { conference:, year: } -> "SIGGRAPH 2024" / "SIGGRAPH Asia 2024"
  def self.conference_label(conference, year)
    "#{conference} #{year}"
  end

  # { conference:, year: } -> "siggraph-2024" / "siggraph-asia-2024"
  def self.conference_slug(conference, year)
    derive_slug(conference_label(conference, year))
  end

  # Group papers by a caller-supplied key selector.
  #
  #   group_by(papers) { |p| p["category"] } # => { key => [papers...] }
  #
  # Returns a Hash mapping each distinct key to the list of papers that yielded
  # it, preserving input order within each group.
  def self.group_by(papers, &key_fn)
    result = {}
    return result if papers.nil?

    papers.each do |paper|
      key = key_fn.call(paper)
      (result[key] ||= []) << paper
    end
    result
  end

  # Serialize a paper_index array to the client search index JSON string.
  #
  # Emits a JSON array of objects with the fields the client search needs:
  #   [{ "title", "authors", "tags", "url", "conference_label", "category",
  #      "track" }]
  #
  # Text search still matches only title/authors/tags; `conference_label`,
  # `category`, and `track` are carried so the client can offer scope filters
  # (search within a conference / category / track). The result is always valid
  # JSON parseable by JSON.parse, and every field survives the round-trip intact
  # (Property 10). `authors`/`tags` are normalized to arrays; missing string
  # fields degrade to "".
  # (Requirements 5.1, 14.2)
  def self.build_search_index(paper_index)
    entries = Array(paper_index).map do |paper|
      {
        "title" => search_string(fetch_field(paper, "title")),
        "authors" => search_list(fetch_field(paper, "authors")),
        "tags" => search_list(fetch_field(paper, "tags")),
        "url" => search_string(fetch_field(paper, "url")),
        "conference_label" => search_string(fetch_field(paper, "conference_label")),
        "category" => search_string(fetch_field(paper, "category")),
        "track" => search_string(fetch_field(paper, "track"))
      }
    end
    JSON.generate(entries)
  end

  # Build the Paper_Index from a list of note front-matter records.
  #
  # Pure validity filter: returns only the records for which `valid_note?`
  # holds (non-blank title, non-empty authors, valid category, valid track).
  # Building the index NEVER raises — a record that would otherwise blow up is
  # simply treated as invalid and excluded, so a single malformed record can
  # never abort the build (Requirement 5.4). Input order is preserved.
  #
  # This is the pure analogue of the generator's per-note validity gate; the
  # generator applies the same `valid_note?` predicate per document.
  # (Requirements 2.2, 2.3, 2.4, 2.5, 3.2, 3.3, 5.1, 5.4, 5.5)
  def self.build_paper_index(records)
    Array(records).select do |record|
      begin
        valid_note?(record)
      rescue StandardError
        false
      end
    end
  end

  # Resolve duplicate slugs, keeping the first-seen entry and dropping any later
  # collision (Requirement 4.4). Returns `{ kept: [...], dropped: [...] }`,
  # preserving input order within each list. Two entries collide iff they share
  # the same value under `slug_key` (string or symbol key tolerated).
  def self.dedupe_by_slug(entries, slug_key = "slug")
    kept = []
    dropped = []
    seen = {}
    Array(entries).each do |entry|
      slug = fetch_field(entry, slug_key)
      if seen.key?(slug)
        dropped << entry
      else
        seen[slug] = true
        kept << entry
      end
    end
    { kept: kept, dropped: dropped }
  end

  # Build the home-page conference cards from the on-disk folder names and the
  # already-filtered list of valid papers.
  #
  # Produces exactly one card per PARSEABLE folder — including folders that hold
  # zero valid papers — as `{ "label", "slug", "count" }`, where `label` is
  # `conference_label(conference, year)`, `slug` is `conference_slug(...)`, and
  # `count` is the number of valid papers whose `conference_label` matches that
  # folder. Unparseable folder names are skipped (the generator warns about
  # them separately). Input folder order is preserved. (Requirements 6.1, 6.2)
  def self.build_conferences(folders, valid_papers)
    conferences = []
    Array(folders).each do |folder|
      parsed = parse_conference_folder(folder)
      next if parsed.nil?

      label = conference_label(parsed[:conference], parsed[:year])
      slug = conference_slug(parsed[:conference], parsed[:year])
      count = Array(valid_papers).count { |paper| fetch_field(paper, "conference_label") == label }

      conferences << { "label" => label, "slug" => slug, "count" => count }
    end
    conferences
  end

  # Empty-state decision for any listing surface (home, conference, category,
  # tag, search results).
  #
  # These two predicates model the mutually-exclusive if/else the listing
  # layouts use: entries render iff the collection is non-empty, the empty-state
  # renders iff the collection is empty, and for any count exactly one of the
  # two is true (they XOR — never both, never neither). This mirrors the Liquid
  # `{% if paper_count > 0 %} ... {% else %} ... {% endif %}` in
  # notes_listing.html / notes_home.html. (Requirements 6.4, 6.5, 7.3, 7.4,
  # 8.3, 8.4, 9.3, 9.4, 14.5, 11.8; design Property 11)
  def self.renders_entries?(count)
    count.to_i.positive?
  end

  # true iff the empty-state should render (collection empty). Complement of
  # renders_entries? for every integer count, so exactly one holds.
  def self.renders_empty_state?(count)
    !renders_entries?(count)
  end

  # --- Internal helpers -------------------------------------------------------

  def self.search_string(value)
    value.is_a?(String) ? value : (value.nil? ? "" : value.to_s)
  end
  private_class_method :search_string

  def self.search_list(value)
    Array(value).map { |item| item.is_a?(String) ? item : item.to_s }
  end
  private_class_method :search_list

  # Read a field from a hash-like front matter, tolerating string or symbol keys.
  def self.fetch_field(front_matter, key)
    if front_matter.respond_to?(:key?) && front_matter.key?(key)
      front_matter[key]
    elsif front_matter.respond_to?(:key?) && front_matter.key?(key.to_sym)
      front_matter[key.to_sym]
    else
      front_matter[key]
    end
  rescue StandardError
    nil
  end
  private_class_method :fetch_field

  def self.non_blank_title?(title)
    title.is_a?(String) && !title.strip.empty?
  end
  private_class_method :non_blank_title?

  def self.non_empty_list?(authors)
    authors.is_a?(Array) && !authors.empty?
  end
  private_class_method :non_empty_list?
end

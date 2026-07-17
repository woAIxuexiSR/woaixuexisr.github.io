# frozen_string_literal: true

require_relative "notes_logic"

# NotesGenerator — the build-time orchestrator for the Paper Notes Database.
#
# This Jekyll::Generator runs after collections are read. It does NOT
# re-implement any parsing / slug / validation logic; all of that lives in the
# pure, dependency-free NotesLogic module. The generator only orchestrates:
#
#   1. Per `notes` document: derive conference / year / conference_label / slug
#      / valid into doc.data (and set the doc slug so the /notes/paper/:slug/
#      permalink resolves), carrying the front-matter `track` through.
#   2. Scan `_notes/` on disk for EVERY conference sub-folder (including folders
#      with zero valid papers) so the home page can show a card per conference
#      (Requirement 6.1).
#   3. Build site.data['paper_index'] (valid-paper hashes) and
#      site.data['conferences'] ({label, slug, count}, incl. empty folders).
#   4. Emit PageWithoutAFile listing pages: one Conference_Page per folder, the
#      seven Category_Pages, and one Tag_Page per distinct tag among valid
#      papers, each with layout `notes_listing` and a pre-filtered `papers` list.
#
# Robustness (Requirement 5.4): per-note processing is wrapped in a rescue so an
# unexpected exception on one file degrades to "excluded + warned" rather than
# aborting the build. Invalid notes, unparseable folders, and duplicate slugs
# are excluded + warned and never fail the build.
#
# Design: "Generator plugin responsibilities", "Error Handling".
module Jekyll
  class NotesGenerator < Generator
    safe false
    priority :normal

    LOG_TOPIC = "NotesGenerator:"
    LISTING_LAYOUT = "notes_listing"

    def generate(site)
      collection = site.collections["notes"]
      return if collection.nil?

      # Phase A — derive fields and validity per document. Returns a candidate
      # (carrying the doc, slug, note path, and its Paper_Index entry) for every
      # note that lives in a parseable folder and passes valid_note?, or nil for
      # excluded notes (which are warned about here).
      candidates = []
      collection.docs.each do |doc|
        candidate = evaluate_doc(doc)
        candidates << candidate if candidate
      rescue StandardError => e
        # Robustness: one bad file must never abort the build (Requirement 5.4).
        doc.data["valid"] = false
        Jekyll.logger.warn(
          LOG_TOPIC,
          "Excluding note #{note_path(doc)} due to unexpected error: #{e.class}: #{e.message}"
        )
      end

      # Phase B — resolve duplicate slugs (keep first-seen) using the pure
      # NotesLogic.dedupe_by_slug helper, then collect the surviving entries.
      valid_papers = resolve_unique(candidates)

      folders = conference_folders(site)

      site.data["paper_index"] = valid_papers
      site.data["conferences"] = build_conferences(folders, valid_papers)

      emit_conference_pages(site, folders, valid_papers)
      emit_category_pages(site, valid_papers)
      emit_tag_pages(site, valid_papers)
    end

    private

    # Derive fields for a single note document. Sets slug / conference / year /
    # conference_label / valid into doc.data and returns a candidate hash
    # (`{ doc:, slug:, note_path:, entry: }`) when the note lives in a parseable
    # folder and passes valid_note?; otherwise warns and returns nil. Duplicate
    # slug resolution happens later, in resolve_unique.
    def evaluate_doc(doc)
      folder = File.basename(File.dirname(doc.relative_path))
      slug = NotesLogic.derive_slug(File.basename(doc.relative_path))

      # Set the slug so the /notes/paper/:slug/ permalink resolves.
      doc.data["slug"] = slug

      parsed = NotesLogic.parse_conference_folder(folder)
      if parsed.nil?
        doc.data["valid"] = false
        # Suppress the collection's auto-generated detail page for an excluded
        # note so it does not become an orphan page with broken category/tag
        # links (Requirement 5.4).
        doc.data["published"] = false
        Jekyll.logger.warn(
          LOG_TOPIC,
          "Unparseable conference folder '#{folder}' for #{note_path(doc)}; excluding note."
        )
        return nil
      end

      conference = parsed[:conference]
      year = parsed[:year]
      label = NotesLogic.conference_label(conference, year)

      doc.data["conference"] = conference
      doc.data["year"] = year
      doc.data["conference_label"] = label
      # `track` is carried through unchanged from the front matter (doc.data["track"]).

      valid = NotesLogic.valid_note?(doc.data)
      doc.data["valid"] = valid

      unless valid
        # Suppress the orphan detail page for an invalid note (Requirement 5.4).
        doc.data["published"] = false
        Jekyll.logger.warn(
          LOG_TOPIC,
          "Invalid note #{note_path(doc)} (needs non-blank title, non-empty authors, " \
          "valid category, and valid track); excluding from index and listings."
        )
        return nil
      end

      {
        doc: doc,
        slug: slug,
        note_path: note_path(doc),
        entry: index_entry(doc, slug, conference, year, label)
      }
    end

    # Resolve duplicate slugs across all candidate notes using the pure
    # NotesLogic.dedupe_by_slug helper: keep the first-seen note for each slug,
    # exclude later collisions (mark them invalid + warn), and return the
    # surviving Paper_Index entries in document order (Requirement 4.4).
    def resolve_unique(candidates)
      resolved = NotesLogic.dedupe_by_slug(candidates, :slug)

      first_seen = {}
      resolved[:kept].each { |candidate| first_seen[candidate[:slug]] = candidate[:note_path] }

      resolved[:dropped].each do |candidate|
        candidate[:doc].data["valid"] = false
        # Suppress the orphan detail page for the excluded duplicate.
        candidate[:doc].data["published"] = false
        Jekyll.logger.warn(
          LOG_TOPIC,
          "Duplicate slug '#{candidate[:slug]}' at #{candidate[:note_path]} collides with " \
          "#{first_seen[candidate[:slug]]}; keeping first-seen, excluding this note."
        )
      end

      resolved[:kept].map { |candidate| candidate[:entry] }
    end

    # Build one Paper_Index entry from a valid document.
    def index_entry(doc, slug, conference, year, label)
      {
        "slug" => slug,
        "title" => doc.data["title"],
        "authors" => Array(doc.data["authors"]),
        "category" => doc.data["category"],
        "track" => doc.data["track"],
        "tags" => Array(doc.data["tags"]),
        "links" => doc.data["links"] || {},
        "conference" => conference,
        "year" => year,
        "conference_label" => label,
        "url" => "/notes/paper/#{slug}/"
      }
    end

    # All conference sub-folders on disk under `_notes/`, including empty ones,
    # sorted for deterministic output.
    def conference_folders(site)
      notes_dir = File.join(site.source, "_notes")
      return [] unless File.directory?(notes_dir)

      Dir.children(notes_dir)
         .select { |entry| File.directory?(File.join(notes_dir, entry)) }
         .sort
    end

    # site.data['conferences'] — one entry per parseable folder (incl. empty),
    # with a count of the valid papers that resolve to that conference label.
    # The card construction and counting is the pure NotesLogic.build_conferences
    # function; the generator only adds the unparseable-folder warnings.
    def build_conferences(folders, valid_papers)
      folders.each do |folder|
        next unless NotesLogic.parse_conference_folder(folder).nil?

        Jekyll.logger.warn(
          LOG_TOPIC,
          "Unparseable conference folder '#{folder}'; omitting from home cards."
        )
      end

      NotesLogic.build_conferences(folders, valid_papers)
    end

    # One Conference_Page per folder at /notes/<conference-slug>/, including
    # empty folders (which render their empty-state) so home cards always link
    # to a generated page (Requirement 6.1, 7.x).
    def emit_conference_pages(site, folders, valid_papers)
      folders.each do |folder|
        parsed = NotesLogic.parse_conference_folder(folder)
        next if parsed.nil?

        label = NotesLogic.conference_label(parsed[:conference], parsed[:year])
        slug = NotesLogic.conference_slug(parsed[:conference], parsed[:year])
        papers = valid_papers.select { |paper| paper["conference_label"] == label }

        add_listing_page(site, "notes/#{slug}", label, papers, "conference")
      end
    end

    # The seven Category_Pages at /notes/category/<category-slug>/.
    def emit_category_pages(site, valid_papers)
      NotesLogic::CATEGORIES.each do |category|
        slug = NotesLogic.derive_slug(category)
        papers = valid_papers.select { |paper| paper["category"] == category }

        add_listing_page(site, "notes/category/#{slug}", category, papers, "category")
      end
    end

    # One Tag_Page per distinct tag among valid papers at
    # /notes/tags/<tag-slug>/. Tags that reduce to the same slug are merged,
    # keeping the first-seen display name.
    def emit_tag_pages(site, valid_papers)
      groups = {}
      valid_papers.each do |paper|
        Array(paper["tags"]).each do |tag|
          slug = NotesLogic.derive_slug(tag.to_s)
          next if slug.empty?

          groups[slug] ||= { name: tag, papers: [] }
          groups[slug][:papers] << paper
        end
      end

      groups.each do |slug, info|
        add_listing_page(site, "notes/tags/#{slug}", info[:name], info[:papers], "tag")
      end
    end

    # Create and register a notes_listing page with a pre-filtered papers list.
    def add_listing_page(site, dir, heading, papers, listing_type)
      page = PageWithoutAFile.new(site, site.source, dir, "index.html")
      page.content = ""
      page.data["layout"] = LISTING_LAYOUT
      page.data["title"] = heading
      page.data["heading"] = heading
      page.data["listing_type"] = listing_type
      page.data["papers"] = papers
      site.pages << page
      page
    end

    def note_path(doc)
      doc.relative_path
    rescue StandardError
      doc.path
    end
  end
end

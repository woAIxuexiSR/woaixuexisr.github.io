# frozen_string_literal: true

source "https://rubygems.org"

gem "jekyll-theme-chirpy", "~> 7.6"

gem "html-proofer", "~> 5.0", group: :test

group :test do
  gem "rspec", "~> 3.13"
  gem "rantly", "~> 2.0"
end

platforms :windows, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:windows]

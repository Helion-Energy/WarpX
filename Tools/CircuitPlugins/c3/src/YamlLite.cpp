/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "YamlLite.H"

#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace c3 {

namespace {

struct Line
{
    int indent;
    std::string text;   // content after the indent
};

std::string
StripQuotes (std::string const& s)
{
    if (s.size() >= 2 && ((s.front() == '\'' && s.back() == '\'')
                          || (s.front() == '"' && s.back() == '"'))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

std::string
Trim (std::string const& s)
{
    std::size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) { return {}; }
    std::size_t e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

/** Split "key: value" / "key:"; returns false if the text is not a
 * mapping entry (no colon separator outside a quoted key). */
bool
SplitKeyValue (std::string const& text, std::string& key, std::string& value)
{
    if (!text.empty() && (text[0] == '\'' || text[0] == '"')) {
        // quoted key ('0101': act101) -- or a quoted scalar, which is
        // not a mapping entry even when it contains a colon
        char const q = text[0];
        std::size_t const close = text.find(q, 1);
        if (close == std::string::npos || close + 1 >= text.size()
            || text[close + 1] != ':') {
            return false;
        }
        key = text.substr(1, close - 1);
        if (close + 2 >= text.size()) {          // "'key':" trailing colon
            value.clear();
            return true;
        }
        if (text[close + 2] != ' ') {
            return false;
        }
        value = Trim(text.substr(close + 3));
        return true;
    }
    // unquoted keys never contain a colon; the first ": " or a
    // trailing ":" separates
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == ':') {
            if (i + 1 == text.size()) {
                key = StripQuotes(Trim(text.substr(0, i)));
                value.clear();
                return true;
            }
            if (text[i + 1] == ' ') {
                key = StripQuotes(Trim(text.substr(0, i)));
                value = Trim(text.substr(i + 1));
                return true;
            }
        }
    }
    return false;
}

/** True when s ends inside-out of a q-quoted scalar: the trailing run of
 * quote characters has odd length (yaml doubles quotes to escape them,
 * so an even run is an escaped quote inside a still-open scalar). */
bool
EndsWithClosingQuote (std::string const& s, char q)
{
    if (s.empty() || s.back() != q) { return false; }
    std::size_t run = 0;
    for (auto it = s.rbegin(); it != s.rend() && *it == q; ++it) {
        run += 1;
    }
    return (run % 2) == 1;
}

class Parser
{
public:
    explicit Parser (std::vector<Line> lines) : m_lines(std::move(lines)) {}

    YamlNode ParseBlock (int indent)
    {
        if (m_cur < m_lines.size() && m_lines[m_cur].indent >= indent
            && m_lines[m_cur].text.rfind("- ", 0) == 0) {
            return ParseSeq(m_lines[m_cur].indent);
        }
        return ParseMap(indent);
    }

private:
    YamlNode ParseSeq (int indent)
    {
        YamlNode node;
        node.m_kind = YamlNode::Kind::Seq;
        while (m_cur < m_lines.size() && m_lines[m_cur].indent == indent
               && m_lines[m_cur].text.rfind("- ", 0) == 0) {
            std::string item = Trim(m_lines[m_cur].text.substr(2));
            std::string key, value;
            if (SplitKeyValue(item, key, value)) {
                // "- key: value" inline-first map item: the rest of the
                // item's keys sit two columns deeper than the dash
                m_lines[m_cur].indent = indent + 2;
                m_lines[m_cur].text = item;
                node.m_seq.push_back(ParseMap(indent + 2));
            } else {
                ++m_cur;
                // multi-line quoted scalar item: consume deeper-indented
                // continuation lines up to the closing quote
                if (!item.empty() && (item[0] == '\'' || item[0] == '"')
                    && !(item.size() >= 2
                         && EndsWithClosingQuote(item, item[0]))) {
                    ConsumeQuoted(item, item[0], indent);
                }
                YamlNode leaf;
                leaf.m_scalar = StripQuotes(item);
                node.m_seq.push_back(std::move(leaf));
            }
        }
        return node;
    }

    YamlNode ParseMap (int indent)
    {
        YamlNode node;
        node.m_kind = YamlNode::Kind::Map;
        while (m_cur < m_lines.size() && m_lines[m_cur].indent == indent) {
            std::string key, value;
            if (!SplitKeyValue(m_lines[m_cur].text, key, value)) {
                throw std::runtime_error("YamlLite: expected 'key: value' at '"
                                         + m_lines[m_cur].text + "'");
            }
            ++m_cur;
            if (!value.empty()) {
                if ((value[0] == '\'' || value[0] == '"')
                    && !(value.size() >= 2
                         && EndsWithClosingQuote(value, value[0]))) {
                    // multi-line quoted value: colons inside are text
                    ConsumeQuoted(value, value[0], indent);
                } else {
                    // plain-scalar continuation lines (deeper indent, not
                    // a sequence dash, no mapping colon) fold in
                    while (m_cur < m_lines.size()
                           && m_lines[m_cur].indent > indent
                           && m_lines[m_cur].text.rfind("- ", 0) != 0) {
                        std::string k2, v2;
                        if (SplitKeyValue(m_lines[m_cur].text, k2, v2)) {
                            break;
                        }
                        value += " " + m_lines[m_cur].text;
                        ++m_cur;
                    }
                }
                YamlNode leaf;
                leaf.m_scalar = StripQuotes(value);
                node.m_map.emplace_back(key, std::move(leaf));
            } else if (m_cur < m_lines.size()
                       && (m_lines[m_cur].indent > indent
                           || (m_lines[m_cur].indent == indent
                               && m_lines[m_cur].text.rfind("- ", 0) == 0))) {
                // nested block (a sequence may sit at the parent indent)
                node.m_map.emplace_back(key,
                                        ParseBlock(m_lines[m_cur].indent));
            } else {
                node.m_map.emplace_back(key, YamlNode{});   // empty scalar
            }
        }
        return node;
    }

    //! fold deeper-indented lines into an open q-quoted scalar until the
    //! closing quote (yaml ''-escapes leave the scalar open)
    void ConsumeQuoted (std::string& value, char q, int indent)
    {
        while (m_cur < m_lines.size() && m_lines[m_cur].indent > indent) {
            value += " " + m_lines[m_cur].text;
            ++m_cur;
            if (EndsWithClosingQuote(value, q)) { break; }
        }
    }

    std::vector<Line> m_lines;
    std::size_t m_cur = 0;
};

} // namespace

bool
YamlNode::Has (std::string const& key) const
{
    for (auto const& kv : m_map) {
        if (kv.first == key) { return true; }
    }
    return false;
}

YamlNode const&
YamlNode::At (std::string const& key) const
{
    for (auto const& kv : m_map) {
        if (kv.first == key) { return kv.second; }
    }
    throw std::runtime_error("YamlLite: missing key '" + key + "'");
}

double
YamlNode::AsDouble () const
{
    char* end = nullptr;
    double const v = std::strtod(m_scalar.c_str(), &end);
    if (end == m_scalar.c_str() || *end != '\0') {
        throw std::runtime_error("YamlLite: non-numeric scalar '"
                                 + m_scalar + "'");
    }
    return v;
}

bool
YamlNode::IsNull () const
{
    return m_kind == Kind::Scalar
        && (m_scalar.empty() || m_scalar == "null" || m_scalar == "~");
}

YamlNode
ParseYamlFile (std::string const& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("YamlLite: cannot open '" + path + "'");
    }
    std::vector<Line> lines;
    std::string raw;
    while (std::getline(in, raw)) {
        if (!raw.empty() && raw.back() == '\r') { raw.pop_back(); }
        std::size_t const ind = raw.find_first_not_of(' ');
        if (ind == std::string::npos) { continue; }          // blank
        if (raw[ind] == '#') { continue; }                   // comment line
        lines.push_back({static_cast<int>(ind), raw.substr(ind)});
    }
    Parser p(std::move(lines));
    return p.ParseBlock(0);
}

} // namespace c3

// SPDX-Licence-Identifier: EUPL-1.2

package ebook

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	defaultWeightChapterChars = 4_000_000
	charsPerPrintedPage       = 2000
	pagesPerVolume            = 300
	euplNotice                = "This work is licensed under the European Union Public Licence v1.2 (EUPL-1.2)."
)

// ModelBookOptions configures BuildModelBook.
type ModelBookOptions struct {
	ModelDir       string
	Title          string
	Author         string
	ForewordPath   string
	IncludeWeights bool
	ChapterChars   int
}

type weightFile struct {
	name     string
	bytes    int
	sha256   string
	tensors  int
	elements int64
	b64      string
}

// BuildModelBook renders a local safetensors model directory as an EPUB book.
// It performs file I/O only; no model runtime is loaded.
func BuildModelBook(opts ModelBookOptions) (*Book, error) {
	if strings.TrimSpace(opts.ModelDir) == "" {
		return nil, fmt.Errorf("ebook: model dir is required")
	}
	entries, err := os.ReadDir(opts.ModelDir)
	if err != nil {
		return nil, fmt.Errorf("ebook: list model dir: %w", err)
	}
	title := opts.Title
	if title == "" {
		title = filepath.Base(opts.ModelDir)
	}
	author := opts.Author
	if author == "" {
		author = "Lethean"
	}
	chapterChars := opts.ChapterChars
	if chapterChars <= 0 {
		chapterChars = defaultWeightChapterChars
	}

	foreword, forewordPath, err := readForeword(opts)
	if err != nil {
		return nil, err
	}
	configJSON := readOptionalText(filepath.Join(opts.ModelDir, "config.json"))
	files, err := readWeightFiles(opts.ModelDir, entries, opts.IncludeWeights)
	if err != nil {
		return nil, err
	}

	book := &Book{Title: title, Author: author, Modified: time.Now().UTC()}
	book.Chapters = append(book.Chapters, titleChapter(title, author, opts.IncludeWeights))
	book.Chapters = append(book.Chapters, forewordChapter(foreword, forewordPath))
	book.Chapters = append(book.Chapters, methodChapter(configJSON, files, opts.IncludeWeights, chapterChars))
	if opts.IncludeWeights {
		book.Chapters = append(book.Chapters, weightChapters(files, chapterChars)...)
	}
	book.Chapters = append(book.Chapters, colophonChapter(files, opts.IncludeWeights))
	return book, nil
}

func readForeword(opts ModelBookOptions) (string, string, error) {
	forewordPath := strings.TrimSpace(opts.ForewordPath)
	if forewordPath == "" {
		candidate := filepath.Join(opts.ModelDir, "README.md")
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			forewordPath = candidate
		}
	}
	if forewordPath == "" {
		return "", "", nil
	}
	data, err := os.ReadFile(forewordPath)
	if err != nil {
		return "", "", fmt.Errorf("ebook: read foreword: %w", err)
	}
	return string(data), forewordPath, nil
}

func readOptionalText(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(data)
}

func readWeightFiles(modelDir string, entries []os.DirEntry, includeWeights bool) ([]weightFile, error) {
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".safetensors") {
			continue
		}
		names = append(names, entry.Name())
	}
	sort.Strings(names)
	if len(names) == 0 {
		return nil, fmt.Errorf("ebook: no .safetensors files found in %s", modelDir)
	}

	files := make([]weightFile, 0, len(names))
	for _, name := range names {
		raw, err := os.ReadFile(filepath.Join(modelDir, name))
		if err != nil {
			return nil, fmt.Errorf("ebook: read %s: %w", name, err)
		}
		sum := sha256.Sum256(raw)
		tensors, elements, _ := safetensorsStats(raw)
		wf := weightFile{
			name:     name,
			bytes:    len(raw),
			sha256:   fmt.Sprintf("%x", sum[:]),
			tensors:  tensors,
			elements: elements,
		}
		if includeWeights {
			wf.b64 = base64.StdEncoding.EncodeToString(raw)
		}
		files = append(files, wf)
	}
	return files, nil
}

func titleChapter(title, author string, weights bool) Chapter {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("<h1>%s</h1>\n", xmlEscape(title)))
	b.WriteString(fmt.Sprintf("<p><em>by %s</em></p>\n", xmlEscape(author)))
	b.WriteString("<hr/>\n")
	if weights {
		b.WriteString("<p>This book is a model. Its later chapters are the model weights rendered as text, which decode back into a runnable model.</p>\n")
	} else {
		b.WriteString("<p>This book describes a model: its foreword, method, and weight inventory. The weights are omitted from this edition.</p>\n")
	}
	b.WriteString(fmt.Sprintf("<p>%s</p>\n", xmlEscape(euplNotice)))
	return Chapter{ID: "ch000-title", Title: title, Body: b.String(), InNav: true}
}

func forewordChapter(foreword, source string) Chapter {
	var b strings.Builder
	b.WriteString("<h1>Foreword</h1>\n")
	if foreword == "" {
		b.WriteString("<p>No foreword was supplied with this model.</p>\n")
	} else {
		if source != "" {
			b.WriteString(fmt.Sprintf("<p><em>From %s.</em></p>\n", xmlEscape(filepath.Base(source))))
		}
		b.WriteString(fmt.Sprintf("<pre>%s</pre>\n", xmlEscape(foreword)))
	}
	return Chapter{ID: "ch001-foreword", Title: "Foreword", Body: b.String(), InNav: true}
}

func methodChapter(configJSON string, files []weightFile, weights bool, chapterChars int) Chapter {
	var b strings.Builder
	b.WriteString("<h1>Method</h1>\n")
	if configJSON != "" {
		b.WriteString("<h2>Architecture</h2>\n")
		b.WriteString(fmt.Sprintf("<pre>%s</pre>\n", xmlEscape(configJSON)))
	}
	b.WriteString("<h2>Inventory</h2>\n<ul>\n")
	var totalBytes, totalB64 int
	var totalTensors int
	var totalElements int64
	for i := range files {
		f := &files[i]
		totalBytes += f.bytes
		totalTensors += f.tensors
		totalElements += f.elements
		totalB64 += len(f.b64)
		shortHash := f.sha256
		if len(shortHash) > 16 {
			shortHash = shortHash[:16]
		}
		b.WriteString(fmt.Sprintf("  <li><code>%s</code> - %s bytes, %d tensors, %s scalars, sha256 %s</li>\n",
			xmlEscape(f.name), grouped(int64(f.bytes)), f.tensors, grouped(f.elements), shortHash))
	}
	b.WriteString("</ul>\n")
	b.WriteString("<h2>This book in numbers</h2>\n<ul>\n")
	b.WriteString(fmt.Sprintf("  <li>%s tensors, %s stored scalars across %d file(s)</li>\n", grouped(int64(totalTensors)), grouped(totalElements), len(files)))
	b.WriteString(fmt.Sprintf("  <li>%s bytes of weights on disk</li>\n", grouped(int64(totalBytes))))
	if weights {
		pages := (totalB64 + charsPerPrintedPage - 1) / charsPerPrintedPage
		volumes := (pages + pagesPerVolume - 1) / pagesPerVolume
		chapters := (totalB64 + chapterChars - 1) / chapterChars
		b.WriteString(fmt.Sprintf("  <li>%s base64 characters of weights, in %d plate(s)</li>\n", grouped(int64(totalB64)), chapters))
		b.WriteString(fmt.Sprintf("  <li>about %s printed pages at %d chars/page, about %s volume(s) of %d pages</li>\n",
			grouped(int64(pages)), charsPerPrintedPage, grouped(int64(volumes)), pagesPerVolume))
	} else {
		b.WriteString("  <li>Weights omitted from this edition.</li>\n")
	}
	b.WriteString("</ul>\n")
	return Chapter{ID: "ch002-method", Title: "Method", Body: b.String(), InNav: true}
}

func weightChapters(files []weightFile, chapterChars int) []Chapter {
	chapters := make([]Chapter, 0, 16)
	var intro strings.Builder
	intro.WriteString("<h1>The Weights</h1>\n")
	intro.WriteString("<p>The following plates are base64-encoded model weights. To reconstruct the model, concatenate each file's plates in order, decode base64, write the named file, and verify sha256.</p>\n<ul>\n")
	for i := range files {
		f := &files[i]
		n := (len(f.b64) + chapterChars - 1) / chapterChars
		intro.WriteString(fmt.Sprintf("  <li><code>%s</code> - %d plate(s), sha256 %s</li>\n", xmlEscape(f.name), n, f.sha256))
	}
	intro.WriteString("</ul>\n")
	chapters = append(chapters, Chapter{ID: "ch003-weights", Title: "The Weights", Body: intro.String(), InNav: true})

	plate := 0
	for i := range files {
		f := &files[i]
		part := 0
		for off := 0; off < len(f.b64); off += chapterChars {
			end := off + chapterChars
			if end > len(f.b64) {
				end = len(f.b64)
			}
			part++
			plate++
			var b strings.Builder
			b.WriteString(fmt.Sprintf("<h2>%s - plate %d</h2>\n", xmlEscape(f.name), part))
			b.WriteString(fmt.Sprintf("<pre>%s</pre>\n", f.b64[off:end]))
			chapters = append(chapters, Chapter{
				ID:    fmt.Sprintf("plate%04d", plate),
				Title: fmt.Sprintf("%s - plate %d", f.name, part),
				Body:  b.String(),
				InNav: false,
			})
		}
	}
	return chapters
}

func colophonChapter(files []weightFile, weights bool) Chapter {
	var b strings.Builder
	b.WriteString("<h1>Colophon</h1>\n")
	b.WriteString(fmt.Sprintf("<p>Generated by <code>lthn-rocm ebook</code> on %s.</p>\n", time.Now().UTC().Format("2 January 2006")))
	b.WriteString("<h2>Provenance</h2>\n<ul>\n")
	for i := range files {
		f := &files[i]
		b.WriteString(fmt.Sprintf("  <li><code>%s</code> - sha256 %s</li>\n", xmlEscape(f.name), f.sha256))
	}
	b.WriteString("</ul>\n")
	if weights {
		b.WriteString("<p>This edition contains the weights and reconstructs into a runnable model.</p>\n")
	}
	b.WriteString(fmt.Sprintf("<h2>Licence</h2>\n<p>%s</p>\n", xmlEscape(euplNotice)))
	return Chapter{ID: "ch999-colophon", Title: "Colophon", Body: b.String(), InNav: true}
}

func safetensorsStats(raw []byte) (tensors int, elements int64, ok bool) {
	if len(raw) < 8 {
		return 0, 0, false
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	if n == 0 || uint64(len(raw)) < 8+n {
		return 0, 0, false
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &header); err != nil {
		return 0, 0, false
	}
	for name, rawValue := range header {
		if name == "__metadata__" {
			continue
		}
		var tensor struct {
			Shape []int64 `json:"shape"`
		}
		if err := json.Unmarshal(rawValue, &tensor); err != nil || len(tensor.Shape) == 0 {
			continue
		}
		count := int64(1)
		for _, dim := range tensor.Shape {
			count *= dim
		}
		tensors++
		elements += count
	}
	return tensors, elements, true
}

func grouped(n int64) string {
	negative := n < 0
	if negative {
		n = -n
	}
	digits := fmt.Sprintf("%d", n)
	var out strings.Builder
	pre := len(digits) % 3
	if pre == 0 {
		pre = 3
	}
	out.WriteString(digits[:pre])
	for i := pre; i < len(digits); i += 3 {
		out.WriteString(",")
		out.WriteString(digits[i : i+3])
	}
	if negative {
		return "-" + out.String()
	}
	return out.String()
}

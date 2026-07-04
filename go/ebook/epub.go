// SPDX-Licence-Identifier: EUPL-1.2

package ebook

import (
	"archive/zip"
	"crypto/sha256"
	"fmt"
	"io"
	"strings"
	"time"
)

const epubMimetype = "application/epub+zip"

// Chapter is one XHTML spine entry in the generated book.
type Chapter struct {
	ID    string
	Title string
	Body  string
	InNav bool
}

// Book is an EPUB3 document ready to stream to an io.Writer.
type Book struct {
	Title      string
	Author     string
	Language   string
	Identifier string
	Rights     string
	Modified   time.Time
	Chapters   []Chapter
}

// WriteEPUB writes a valid EPUB3 zip container. The mimetype entry is first and
// stored uncompressed, as EPUB readers expect.
func (b *Book) WriteEPUB(w io.Writer) error {
	if len(b.Chapters) == 0 {
		return fmt.Errorf("ebook: a book needs at least one chapter")
	}
	lang := defaultString(b.Language, "en")
	rights := defaultString(b.Rights, "EUPL-1.2")
	id := b.Identifier
	if id == "" {
		sum := sha256.Sum256([]byte(b.Title + "\x00" + b.Author))
		id = fmt.Sprintf("urn:lethean:ebook:%x", sum[:8])
	}
	modified := b.Modified
	if modified.IsZero() {
		modified = time.Now().UTC()
	}

	zw := zip.NewWriter(w)
	mw, err := zw.CreateHeader(&zip.FileHeader{Name: "mimetype", Method: zip.Store})
	if err != nil {
		return fmt.Errorf("ebook: create mimetype: %w", err)
	}
	if _, err := io.WriteString(mw, epubMimetype); err != nil {
		return fmt.Errorf("ebook: write mimetype: %w", err)
	}
	for _, entry := range []struct {
		name string
		body string
	}{
		{name: "META-INF/container.xml", body: epubContainerXML},
		{name: "OEBPS/content.opf", body: b.opf(id, lang, rights, modified)},
		{name: "OEBPS/nav.xhtml", body: b.navXHTML()},
	} {
		if err := epubWrite(zw, entry.name, entry.body); err != nil {
			return err
		}
	}
	for i := range b.Chapters {
		ch := &b.Chapters[i]
		if err := epubWrite(zw, "OEBPS/"+ch.ID+".xhtml", chapterXHTML(ch)); err != nil {
			return err
		}
	}
	if err := zw.Close(); err != nil {
		return fmt.Errorf("ebook: finalize epub: %w", err)
	}
	return nil
}

func epubWrite(zw *zip.Writer, name, content string) error {
	fw, err := zw.Create(name)
	if err != nil {
		return fmt.Errorf("ebook: create %s: %w", name, err)
	}
	if _, err := io.WriteString(fw, content); err != nil {
		return fmt.Errorf("ebook: write %s: %w", name, err)
	}
	return nil
}

const epubContainerXML = `<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
`

func (b *Book) opf(id, lang, rights string, modified time.Time) string {
	var out strings.Builder
	out.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	out.WriteString(`<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">` + "\n")
	out.WriteString("  <metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n")
	out.WriteString(fmt.Sprintf("    <dc:identifier id=\"bookid\">%s</dc:identifier>\n", xmlEscape(id)))
	out.WriteString(fmt.Sprintf("    <dc:title>%s</dc:title>\n", xmlEscape(b.Title)))
	out.WriteString(fmt.Sprintf("    <dc:creator>%s</dc:creator>\n", xmlEscape(b.Author)))
	out.WriteString(fmt.Sprintf("    <dc:language>%s</dc:language>\n", xmlEscape(lang)))
	out.WriteString(fmt.Sprintf("    <dc:rights>%s</dc:rights>\n", xmlEscape(rights)))
	out.WriteString(fmt.Sprintf("    <meta property=\"dcterms:modified\">%s</meta>\n", modified.UTC().Format("2006-01-02T15:04:05Z")))
	out.WriteString("  </metadata>\n  <manifest>\n")
	out.WriteString("    <item id=\"nav\" href=\"nav.xhtml\" media-type=\"application/xhtml+xml\" properties=\"nav\"/>\n")
	for i := range b.Chapters {
		ch := &b.Chapters[i]
		out.WriteString(fmt.Sprintf("    <item id=\"%s\" href=\"%s.xhtml\" media-type=\"application/xhtml+xml\"/>\n", ch.ID, ch.ID))
	}
	out.WriteString("  </manifest>\n  <spine>\n")
	for i := range b.Chapters {
		out.WriteString(fmt.Sprintf("    <itemref idref=\"%s\"/>\n", b.Chapters[i].ID))
	}
	out.WriteString("  </spine>\n</package>\n")
	return out.String()
}

func (b *Book) navXHTML() string {
	var out strings.Builder
	out.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	out.WriteString(`<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">` + "\n")
	out.WriteString("<head><title>Contents</title></head>\n<body>\n  <nav epub:type=\"toc\" id=\"toc\">\n    <h1>Contents</h1>\n    <ol>\n")
	for i := range b.Chapters {
		ch := &b.Chapters[i]
		if ch.InNav {
			out.WriteString(fmt.Sprintf("      <li><a href=\"%s.xhtml\">%s</a></li>\n", ch.ID, xmlEscape(ch.Title)))
		}
	}
	out.WriteString("    </ol>\n  </nav>\n</body>\n</html>\n")
	return out.String()
}

func chapterXHTML(ch *Chapter) string {
	var out strings.Builder
	out.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	out.WriteString(`<html xmlns="http://www.w3.org/1999/xhtml">` + "\n")
	out.WriteString(fmt.Sprintf("<head><title>%s</title></head>\n<body>\n", xmlEscape(ch.Title)))
	out.WriteString(ch.Body)
	out.WriteString("\n</body>\n</html>\n")
	return out.String()
}

func xmlEscape(s string) string {
	replacer := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;")
	return replacer.Replace(s)
}

func defaultString(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

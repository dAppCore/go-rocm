// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"dappco.re/go/rocm/ebook"
)

type ebookCommandOptions struct {
	ModelPath      string
	OutputPath     string
	Title          string
	Author         string
	ForewordPath   string
	IncludeWeights bool
	ChapterChars   int
}

type ebookCommandReport struct {
	Version        int               `json:"version"`
	Kind           string            `json:"kind"`
	Backend        string            `json:"backend"`
	Command        string            `json:"command"`
	CLIContract    string            `json:"cli_contract"`
	NoPython       bool              `json:"no_python"`
	ModelPath      string            `json:"model_path"`
	OutputPath     string            `json:"output_path"`
	Title          string            `json:"title"`
	Author         string            `json:"author"`
	IncludeWeights bool              `json:"include_weights"`
	ChapterChars   int               `json:"chapter_chars"`
	Chapters       int               `json:"chapters"`
	NavChapters    int               `json:"nav_chapters"`
	Bytes          int64             `json:"bytes"`
	Labels         map[string]string `json:"labels,omitempty"`
	Notes          []string          `json:"notes,omitempty"`
}

func ebookCommandReportFromOptions(opts ebookCommandOptions) (ebookCommandReport, error) {
	modelPath := strings.TrimSpace(opts.ModelPath)
	if modelPath == "" {
		return ebookCommandReport{}, fmt.Errorf("model path is required")
	}
	info, err := os.Stat(modelPath)
	if err != nil {
		return ebookCommandReport{}, fmt.Errorf("model %s: %w", modelPath, err)
	}
	if !info.IsDir() {
		return ebookCommandReport{}, fmt.Errorf("model %s is not a directory", modelPath)
	}
	outputPath := strings.TrimSpace(opts.OutputPath)
	if outputPath == "" {
		outputPath = filepath.Base(modelPath) + ".epub"
	}
	book, err := ebook.BuildModelBook(ebook.ModelBookOptions{
		ModelDir:       modelPath,
		Title:          opts.Title,
		Author:         opts.Author,
		ForewordPath:   opts.ForewordPath,
		IncludeWeights: opts.IncludeWeights,
		ChapterChars:   opts.ChapterChars,
	})
	if err != nil {
		return ebookCommandReport{}, err
	}
	out, err := os.Create(outputPath)
	if err != nil {
		return ebookCommandReport{}, fmt.Errorf("create %s: %w", outputPath, err)
	}
	writeErr := book.WriteEPUB(out)
	closeErr := out.Close()
	if writeErr != nil {
		return ebookCommandReport{}, writeErr
	}
	if closeErr != nil {
		return ebookCommandReport{}, fmt.Errorf("close %s: %w", outputPath, closeErr)
	}
	size := int64(0)
	if outputInfo, err := os.Stat(outputPath); err == nil {
		size = outputInfo.Size()
	}
	navChapters := 0
	for _, chapter := range book.Chapters {
		if chapter.InNav {
			navChapters++
		}
	}
	chapterChars := opts.ChapterChars
	if chapterChars <= 0 {
		chapterChars = 4_000_000
	}
	return ebookCommandReport{
		Version:        1,
		Kind:           "model-ebook",
		Backend:        defaultBackendName,
		Command:        "ebook",
		CLIContract:    cliContractName,
		NoPython:       true,
		ModelPath:      modelPath,
		OutputPath:     outputPath,
		Title:          book.Title,
		Author:         book.Author,
		IncludeWeights: opts.IncludeWeights,
		ChapterChars:   chapterChars,
		Chapters:       len(book.Chapters),
		NavChapters:    navChapters,
		Bytes:          size,
		Labels: map[string]string{
			"backend":                      defaultBackendName,
			"cli_contract":                 cliContractName,
			"ebook_runtime":                "native_epub3",
			"no_python":                    "true",
			"production_requires_env_gate": "false",
			"production_requires_cli_flag": "false",
		},
		Notes: []string{
			"ROCm ebook generation is pure file I/O and does not load model weights into a runtime.",
		},
	}, nil
}

func printEbookCommandReport(stdout io.Writer, report ebookCommandReport) {
	fmt.Fprintf(stdout, "wrote %s - %d chapters (%d in contents)\n", report.OutputPath, report.Chapters, report.NavChapters)
	fmt.Fprintf(stdout, "epub size %d bytes\n", report.Bytes)
}

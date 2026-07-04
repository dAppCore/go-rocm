// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"encoding/json"
	"io"
	"net/http"

	"dappco.re/go/rocm/score"
)

const (
	rocmServeScorePath = "/v1/score"
	scoreBodyLimit     = 256 * 1024
)

type scoreRequest struct {
	Prompt   string `json:"prompt"`
	Response string `json:"response"`
}

func handleROCmScorePair(w http.ResponseWriter, r *http.Request) {
	if !rocmServeRequireMethod(w, r, http.MethodPost) {
		return
	}
	defer r.Body.Close()
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, scoreBodyLimit))
	if err != nil {
		writeROCmServeError(w, http.StatusBadRequest, "body unreadable or over 256KB", "body")
		return
	}
	var req scoreRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeROCmServeError(w, http.StatusBadRequest, `invalid JSON: expected {"prompt":string,"response":string}`, "body")
		return
	}
	writeROCmServeJSON(w, http.StatusOK, score.ScorePair(req.Prompt, req.Response))
}

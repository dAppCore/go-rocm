//go:build linux && amd64

package rocm

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func exampleModel() *rocmModel {
	return &rocmModel{modelType: "llama", modelInfo: inference.ModelInfo{Architecture: "llama"}}
}
func ExampleModel_Generate() { core.Println(exampleModel().Generate != nil) /* Output: true */ }
func ExampleModel_Chat()     { core.Println(exampleModel().Chat != nil) /* Output: true */ }
func ExampleModel_Classify() { core.Println(exampleModel().Classify != nil) /* Output: true */ }
func ExampleModel_BatchGenerate() {
	core.Println(exampleModel().BatchGenerate != nil) /* Output: true */
}
func ExampleModel_ModelType() { core.Println(exampleModel().ModelType()) /* Output: llama */ }
func ExampleModel_Info()      { core.Println(exampleModel().Info().Architecture) /* Output: llama */ }
func ExampleModel_Metrics()   { core.Println(exampleModel().Metrics().GeneratedTokens) /* Output: 0 */ }
func ExampleModel_Err()       { core.Println(exampleModel().Err() == nil) /* Output: true */ }
func ExampleModel_Close() {
	core.Println((&rocmModel{server: &server{processCommand: &core.Cmd{}}}).Close() == nil) /* Output: true */
}

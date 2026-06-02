# RelayX Go SDK

Go client for the [RelayX](https://relayx.vexarr.com) OpenAI-compatible AI gateway.

```go
client := relayx.NewClient("rx_...")
var out map[string]any
err := client.ChatCompletions(ctx, relayx.ChatCompletionRequest{
    Model:    "gpt-4o",
    Messages: []relayx.ChatMessage{{Role: "user", Content: "Hello from RelayX"}},
}, &out)
```

## Test

```bash
go test ./...
```

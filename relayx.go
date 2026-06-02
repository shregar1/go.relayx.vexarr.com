package relayx

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	Headers    map[string]string
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type RelayXMeta struct {
	Provider      string `json:"provider"`
	ProviderModel string `json:"provider_model"`
	Route         string `json:"route"`
	Cache         string `json:"cache"`
}

type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   TokenUsage   `json:"usage"`
	RelayX  *RelayXMeta  `json:"relayx,omitempty"`
}

type CompletionChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   TokenUsage         `json:"usage"`
}

type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      bool          `json:"stream,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
}

type EmbeddingRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"`
}

type ResponsesRequest struct {
	Model  string      `json:"model"`
	Input  interface{} `json:"input"`
	Stream bool        `json:"stream,omitempty"`
}

type CompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	Stream      bool     `json:"stream,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
}

type Error struct {
	Status int
	Body   string
}

func (e *Error) Error() string {
	return fmt.Sprintf("relayx request failed with status %d", e.Status)
}

type SSEStream struct {
	scanner *bufio.Scanner
	body    io.ReadCloser
}

func (s *SSEStream) Next() (json.RawMessage, bool) {
	for s.scanner.Scan() {
		line := s.scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := line[6:]
		if strings.TrimSpace(payload) == "[DONE]" {
			return nil, false
		}
		return json.RawMessage(payload), true
	}
	return nil, false
}

func (s *SSEStream) Close() error {
	return s.body.Close()
}

func NewClient(apiKey string) *Client {
	return &Client{
		APIKey:  apiKey,
		BaseURL: "https://gateway.relayx.vexarr.com",
		HTTPClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (c *Client) ChatCompletions(ctx context.Context, request ChatCompletionRequest) (*ChatCompletionResponse, error) {
	var out ChatCompletionResponse
	if err := c.postJSON(ctx, "/v1/chat/completions", request, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) Completions(ctx context.Context, request CompletionRequest) (*CompletionResponse, error) {
	var out CompletionResponse
	if err := c.postJSON(ctx, "/v1/completions", request, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) Embeddings(ctx context.Context, request EmbeddingRequest) (*EmbeddingResponse, error) {
	var out EmbeddingResponse
	if err := c.postJSON(ctx, "/v1/embeddings", request, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *Client) Responses(ctx context.Context, request ResponsesRequest, out interface{}) error {
	return c.postJSON(ctx, "/v1/responses", request, out)
}

func (c *Client) Models(ctx context.Context, out interface{}) error {
	return c.doJSON(ctx, http.MethodGet, "/v1/models", nil, out)
}

func (c *Client) StreamChatCompletions(ctx context.Context, request ChatCompletionRequest) (*SSEStream, error) {
	request.Stream = true
	return c.postSSE(ctx, "/v1/chat/completions", request)
}

func (c *Client) StreamResponses(ctx context.Context, request ResponsesRequest) (*SSEStream, error) {
	request.Stream = true
	return c.postSSE(ctx, "/v1/responses", request)
}

func (c *Client) postJSON(ctx context.Context, path string, body interface{}, out interface{}) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	return c.doJSON(ctx, http.MethodPost, path, payload, out)
}

func (c *Client) doJSON(ctx context.Context, method string, path string, body []byte, out interface{}) error {
	baseURL := strings.TrimRight(c.BaseURL, "/")
	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, baseURL+path, bodyReader)
	if err != nil {
		return err
	}

	req.Header.Set("authorization", "Bearer "+c.APIKey)
	req.Header.Set("content-type", "application/json")
	req.Header.Set("x-relayx-sdk", "go")
	for key, value := range c.Headers {
		req.Header.Set(key, value)
	}

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return &Error{Status: resp.StatusCode, Body: string(respBody)}
	}

	if out == nil || len(respBody) == 0 {
		return nil
	}
	return json.Unmarshal(respBody, out)
}

func (c *Client) postSSE(ctx context.Context, path string, body interface{}) (*SSEStream, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	baseURL := strings.TrimRight(c.BaseURL, "/")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}

	req.Header.Set("authorization", "Bearer "+c.APIKey)
	req.Header.Set("content-type", "application/json")
	req.Header.Set("x-relayx-sdk", "go")
	for key, value := range c.Headers {
		req.Header.Set(key, value)
	}

	client := c.HTTPClient
	if client == nil {
		client = &http.Client{}
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		return nil, &Error{Status: resp.StatusCode, Body: string(respBody)}
	}

	return &SSEStream{
		scanner: bufio.NewScanner(resp.Body),
		body:    resp.Body,
	}, nil
}

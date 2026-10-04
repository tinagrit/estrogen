use std::{env, net::SocketAddr, path::PathBuf};

use axum::{
    body::{to_bytes, Body},
    extract::State,
    http::{header, Request, Response, StatusCode},
    routing::any,
    Router,
};
use tower_http::{
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};

const MAX_REQUEST_BYTES: usize = 25 * 1024 * 1024;

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    model_api_url: String,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "eralpha_web=info,tower_http=info".into()),
        )
        .init();

    let model_api_url = env::var("MODEL_API_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8000".to_owned())
        .trim_end_matches('/')
        .to_owned();
    let port = env::var("PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(3000);
    let web_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../web");
    let index = web_root.join("index.html");

    let state = AppState {
        client: reqwest::Client::new(),
        model_api_url: model_api_url.clone(),
    };
    let static_files = ServeDir::new(web_root).not_found_service(ServeFile::new(index));
    let app = Router::new()
        .route("/api", any(proxy_to_model_api))
        .route("/api/{*path}", any(proxy_to_model_api))
        .fallback_service(static_files)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let address = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("could not bind the web server port");
    tracing::info!(%address, %model_api_url, "ERalpha web server ready");
    axum::serve(listener, app)
        .await
        .expect("web server stopped unexpectedly");
}

async fn proxy_to_model_api(
    State(state): State<AppState>,
    request: Request<Body>,
) -> Response<Body> {
    let (parts, body) = request.into_parts();
    let upstream_url = format!("{}{}", state.model_api_url, parts.uri);
    let mut upstream = state.client.request(parts.method, upstream_url);

    for name in [header::CONTENT_TYPE, header::ACCEPT] {
        if let Some(value) = parts.headers.get(&name) {
            upstream = upstream.header(name, value);
        }
    }

    let body = match to_bytes(body, MAX_REQUEST_BYTES).await {
        Ok(body) => body,
        Err(error) => {
            return json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Could not read request body: {error}"),
            )
        }
    };

    let upstream_response = match upstream.body(body).send().await {
        Ok(response) => response,
        Err(error) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                format!("The Python model API is unavailable: {error}"),
            )
        }
    };

    let status = upstream_response.status();
    let response_headers = upstream_response.headers().clone();
    let response_body = match upstream_response.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                format!("Could not read the model API response: {error}"),
            )
        }
    };

    let mut response = Response::builder().status(status);
    for name in [
        header::CONTENT_TYPE,
        header::CONTENT_DISPOSITION,
        header::CACHE_CONTROL,
    ] {
        if let Some(value) = response_headers.get(&name) {
            response = response.header(name, value);
        }
    }
    response
        .body(Body::from(response_body))
        .unwrap_or_else(|_| json_error(StatusCode::BAD_GATEWAY, "Invalid upstream response"))
}

fn json_error(status: StatusCode, message: impl Into<String>) -> Response<Body> {
    let payload = serde_json::json!({ "detail": message.into() }).to_string();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(payload))
        .expect("static JSON response is valid")
}

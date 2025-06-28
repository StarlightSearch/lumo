use actix_web::body::EitherBody;
use actix_web::dev::{ServiceResponse, Transform};
use actix_web::http::header;
use actix_web::{dev::ServiceRequest, Error, HttpResponse};
use futures::TryFutureExt;
use serde_json::json;
use std::future::{ready, Future};
use std::pin::Pin;
use std::task::{Context, Poll};

pub struct ApiKeyAuth;

impl ApiKeyAuth {
    fn validate_api_key(req: &ServiceRequest) -> Result<bool, Error> {
        let api_key = std::env::var("LUMO_API_KEY").map_err(|_| {
            actix_web::error::ErrorInternalServerError("Server API key not configured")
        })?;

        let auth_header = req.headers().get(header::AUTHORIZATION);
        match auth_header {
            Some(auth) => {
                let auth_str = auth.to_str().map_err(|_| {
                    actix_web::error::ErrorBadRequest("Invalid authorization header")
                })?;
                Ok(auth_str == format!("Bearer {}", api_key))
            }
            None => Ok(false),
        }
    }

    fn is_auth_enabled() -> bool {
        std::env::var("ENABLE_AUTH")
            .map(|v| v == "true")
            .unwrap_or(false)
    }
}

impl<S, B> Transform<S, ServiceRequest> for ApiKeyAuth
where
    S: actix_web::dev::Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B>>;
    type Error = Error;
    type Transform = ApiKeyAuthMiddleware<S>;
    type InitError = ();
    type Future = std::future::Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(ApiKeyAuthMiddleware { service }))
    }
}

pub struct ApiKeyAuthMiddleware<S> {
    service: S,
}

impl<S, B> actix_web::dev::Service<ServiceRequest> for ApiKeyAuthMiddleware<S>
where
    S: actix_web::dev::Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<EitherBody<B>>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(&self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        // Skip auth for health check endpoint
        if req.path() == "/health_check" {
            return Box::pin(
                self.service
                    .call(req)
                    .map_ok(|res| res.map_into_left_body()),
            );
        }

        // If auth is disabled, pass through all requests
        if !ApiKeyAuth::is_auth_enabled() {
            return Box::pin(
                self.service
                    .call(req)
                    .map_ok(|res| res.map_into_left_body()),
            );
        }

        // Validate API key
        match ApiKeyAuth::validate_api_key(&req) {
            Ok(true) => Box::pin(
                self.service
                    .call(req)
                    .map_ok(|res| res.map_into_left_body()),
            ),
            Ok(false) => {
                let (http_req, _payload) = req.into_parts();
                let response = HttpResponse::Unauthorized().json(json!({
                    "error": "Invalid or missing API key"
                }));
                let srv_resp = ServiceResponse::new(http_req, response).map_into_right_body();
                Box::pin(ready(Ok(srv_resp)))
            }
            Err(e) => {
                let (http_req, _payload) = req.into_parts();
                let response = HttpResponse::InternalServerError().json(json!({
                    "error": e.to_string()
                }));
                let srv_resp = ServiceResponse::new(http_req, response).map_into_right_body();
                Box::pin(ready(Ok(srv_resp)))
            }
        }
    }
}

use std::net::TcpListener;

use lumo_server::run;

fn spawn_app() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind address");
    let port = listener.local_addr().unwrap().port();
    let server = run(listener).expect("Failed to bind address");
    let _ = tokio::spawn(server);
    format!("http://localhost:{}", port)
}

#[actix_web::test]
async fn health_check_works() {
    let url = spawn_app();
    let client = reqwest::Client::new();
    let response = client
        .get(url + "/health_check")
        .send()
        .await
        .expect("Failed to send request");
    assert!(response.status().is_success());
    assert_eq!(Some(0), response.content_length());
}

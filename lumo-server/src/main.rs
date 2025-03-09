use std::net::TcpListener;

use lumo_server::run;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8080")?;
    run(listener).expect("Failed to bind address").await
}

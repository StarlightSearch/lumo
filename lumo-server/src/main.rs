use std::net::TcpListener;

use lumo_server::run;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let listener = TcpListener::bind("0.0.0.0:8080")?;
    println!("Listening on 0.0.0.0:8080");
    run(listener).expect("Failed to bind address").await
}

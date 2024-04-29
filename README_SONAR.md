## Push to sonarquebe
sonar-scanner \
  -Dsonar.projectKey=knowledge-source \
  -Dsonar.sources=. \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.token=sqp_540a81fa6c8776198485ff1fff93c30161096a3b
